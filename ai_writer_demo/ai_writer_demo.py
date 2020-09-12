"""
DOCSTRING
"""
import caffe
import collections
import config
import copy
import cPickle
import lasagne
import nltk
import numpy
import PIL
import scipy.linalg
import skimage.transform
import search
import sys
import theano

class Base:
    """
    DOCSTRING
    """
    def _p(self, pp, name):
        """
        make prefix-appended name
        """
        return '%s_%s'%(pp, name)

    def fflayer(
        self,
        tparams,
        state_below,
        options,
        prefix='rconv',
        activ='lambda x: tensor.tanh(x)',
        **kwargs):
        """
        Feedforward pass
        """
        return eval(activ)(theano.tensor.dot(
            state_below, tparams[self._p(prefix,'W')])+tparams[self._p(prefix,'b')])

    def get_layer(self, name):
        """
        DOCSTRING
        """
        fns = layers[name]
        return (eval(fns[0]), eval(fns[1]))

    def init_tparams(self, params):
        """
        initialize Theano shared variables according to the initial parameters
        """
        tparams = collections.OrderedDict()
        for kk, pp in params.iteritems():
            tparams[kk] = theano.shared(params[kk], name=kk)
            return tparams
    
    def linear(self, x):
        """
        Linear activation function
        """
        return x

    def load_params(self, path, params):
        """
        load parameters
        """
        pp = numpy.load(path)
        for kk, vv in params.iteritems():
            if kk not in pp:
                warnings.warn('%s is not in the archive'%kk)
                continue
            params[kk] = pp[kk]
        return params
    
    def norm_weight(self, nin, nout=None, scale=0.1, ortho=True):
        """
        Uniform initalization from [-scale, scale]
        If matrix is square and ortho=True, use ortho instead
        """
        if nout == None:
            nout = nin
        if nout == nin and ortho:
            W = self.ortho_weight(nin)
        else:
            W = numpy.random.uniform(low=-scale, high=scale, size=(nin, nout))
        return W.astype('float32')

    def ortho_weight(self, ndim):
        """
        Orthogonal weight init, for recurrent layers
        """
        W = numpy.random.randn(ndim, ndim)
        u, s, v = numpy.linalg.svd(W)
        return u.astype('float32')

    def param_init_gru(self, options, params, prefix='gru', nin=None, dim=None):
        """
        Gated Recurrent Unit (GRU)
        """
        if nin == None:
            nin = options['dim_proj']
        if dim == None:
            dim = options['dim_proj']
        W = numpy.concatenate([self.norm_weight(nin,dim), self.norm_weight(nin,dim)], axis=1)
        params[self._p(prefix,'W')] = W
        params[self._p(prefix,'b')] = numpy.zeros((2 * dim,)).astype('float32')
        U = numpy.concatenate([self.ortho_weight(dim), self.ortho_weight(dim)], axis=1)
        params[self._p(prefix,'U')] = U
        Wx = self.norm_weight(nin, dim)
        params[self._p(prefix,'Wx')] = Wx
        Ux = self.ortho_weight(dim)
        params[self._p(prefix,'Ux')] = Ux
        params[self._p(prefix,'bx')] = numpy.zeros((dim,)).astype('float32')
        return params

class Decoder(Base):
    """
    Decoder
    """
    def __init__(self):
        layers = {
            'ff': ('param_init_fflayer', 'fflayer'),
            'gru': ('param_init_gru', 'gru_layer')}

    def build_sampler(self, tparams, options, trng):
        """
        Forward sampling
        """
        ctx = theano.tensor.matrix('ctx', dtype='float32')
        ctx0 = ctx
        init_state = self.get_layer('ff')[1](
            tparams, ctx, options, prefix='ff_state', activ='tanh')
        f_init = theano.function([ctx], init_state, name='f_init', profile=False)
        y = theano.tensor.vector('y_sampler', dtype='int64')
        init_state = theano.tensor.matrix('init_state', dtype='float32')
        emb = theano.tensor.switch(y[:,None] < 0, theano.tensor.alloc(
            0.0, 1, tparams['Wemb'].shape[1]), tparams['Wemb'][y])
        proj = self.get_layer(options['decoder'])[1](
            tparams, emb, init_state, options, prefix='decoder', mask=None, one_step=True)
        next_state = proj[0]
        if options['doutput']:
            hid = self.get_layer('ff')[1](
                tparams, next_state, options, prefix='ff_hid', activ='tanh')
            logit = self.get_layer('ff')[1](
                tparams, hid, options, prefix='ff_logit', activ='linear')
        else:
            logit = self.get_layer('ff')[1](
                tparams, next_state, options, prefix='ff_logit', activ='linear')
        next_probs = theano.tensor.nnet.softmax(logit)
        next_sample = trng.multinomial(pvals=next_probs).argmax(1)
        inps = [y, init_state]
        outs = [next_probs, next_sample, next_state]
        f_next = theano.function(inps, outs, name='f_next', profile=False)
        return f_init, f_next

    def gru_layer(
        self, 
        tparams,
        state_below,
        init_state,
        options,
        prefix='gru',
        mask=None,
        one_step=False,
        **kwargs):
        """
        Feedforward pass through GRU
        """
        nsteps = state_below.shape[0]
        if state_below.ndim == 3:
            n_samples = state_below.shape[1]
        else:
            n_samples = 1
        dim = tparams[self._p(prefix,'Ux')].shape[1]
        if init_state == None:
            init_state = theano.tensor.alloc(0.0, n_samples, dim)
        if mask == None:
            mask = theano.tensor.alloc(1., state_below.shape[0], 1)

        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n*dim:(n+1)*dim]
            return _x[:, n*dim:(n+1)*dim]

        state_below_ = theano.tensor.dot(
            state_below, tparams[self._p(prefix, 'W')]) + tparams[self._p(prefix, 'b')]
        state_belowx = theano.tensor.dot(
            state_below, tparams[self._p(prefix, 'Wx')]) + tparams[self._p(prefix, 'bx')]
        U = tparams[self._p(prefix, 'U')]
        Ux = tparams[self._p(prefix, 'Ux')]

        def _step_slice(m_, x_, xx_, h_, U, Ux):
            preact = theano.tensor.dot(h_, U)
            preact += x_
            r = theano.tensor.nnet.sigmoid(_slice(preact, 0, dim))
            u = theano.tensor.nnet.sigmoid(_slice(preact, 1, dim))
            preactx = theano.tensor.dot(h_, Ux)
            preactx = preactx * r
            preactx = preactx + xx_
            h = theano.tensor.tanh(preactx)
            h = u * h_ + (1. - u) * h
            h = m_[:,None] * h + (1. - m_)[:,None] * h_
            return h

        seqs = [mask, state_below_, state_belowx]
        _step = _step_slice
        if one_step:
            rval = _step(*(seqs + [
                init_state,
                tparams[self._p(prefix, 'U')],
                tparams[self._p(prefix, 'Ux')]]))
        else:
            rval, updates = theano.scan(
                _step, sequences=seqs, outputs_info=[init_state],
                non_sequences=[tparams[self._p(prefix, 'U')], tparams[self._p(prefix, 'Ux')]],
                name=self._p(prefix, '_layers'), n_steps=nsteps, profile=False, strict=True)
        rval = [rval]
        return rva1

    def init_params(self, options):
        """
        Initialize all parameters
        """
        params = collections.OrderedDict()
        params['Wemb'] = self.norm_weight(options['n_words'], options['dim_word'])
        params = self.get_layer('ff')[0](
            options, params, prefix='ff_state',
            nin=options['dimctx'], nout=options['dim'])
        params = self.get_layer(options['decoder'])[0](
            options, params, prefix='decoder',
            nin=options['dim_word'], dim=options['dim'])
        if options['doutput']:
            params = self.get_layer('ff')[0](
                options, params, prefix='ff_hid',
                nin=options['dim'], nout=options['dim_word'])
            params = self.get_layer('ff')[0](
                options, params, prefix='ff_logit',
                nin=options['dim_word'], nout=options['n_words'])
        else:
            params = self.get_layer('ff')[0](
                options, params, prefix='ff_logit',
                nin=options['dim'], nout=options['n_words'])
        return params

    def load_model(self, path_to_model, path_to_dictionary):
        """
        Load a trained model for decoding
        """
        with open(path_to_dictionary, 'rb') as f:
            worddict = cPickle.load(f)
        # create inverted dictionary
        word_idict = dict()
        for kk, vv in worddict.iteritems():
            word_idict[vv] = kk
        word_idict[0] = '<eos>'
        word_idict[1] = 'UNK'
        # load model options
        with open('%s.cPickle'%path_to_model, 'rb') as f:
            options = cPickle.load(f)
        if 'doutput' not in options.keys():
            options['doutput'] = True
        # load parameters
        params = self.init_params(options)
        params = self.load_params(path_to_model, params)
        tparams = self.init_tparams(params)
        # sampler
        trng = theano.sandbox.rng_mrg.MRG_RandomStreams(1234)
        f_init, f_next = self.build_sampler(tparams, options, trng)
        # pack everything up
        dec = dict()
        dec['options'] = options
        dec['trng'] = trng
        dec['worddict'] = worddict
        dec['word_idict'] = word_idict
        dec['tparams'] = tparams
        dec['f_init'] = f_init
        dec['f_next'] = f_next
        return dec

    def param_init_fflayer(
        self,
        options,
        params,
        prefix='ff',
        nin=None,
        nout=None,
        ortho=True):
        """
        Feedforward layer

        Affine transformation + point-wise nonlinearity
        """
        if nin == None:
            nin = options['dim_proj']
        if nout == None:
            nout = options['dim_proj']
        params[self._p(prefix,'W')] = self.norm_weight(nin, nout)
        params[self._p(prefix,'b')] = numpy.zeros((nout,)).astype('float32')
        return params

    def run_sampler(self, dec, c, beam_width=1, stochastic=False, use_unk=False):
        """
        Generate text conditioned on c
        """
        sample, score = search.gen_sample(
            dec['tparams'], dec['f_init'], dec['f_next'],
            c.reshape(1, dec['options']['dimctx']), dec['options'],
            trng=dec['trng'], k=beam_width, maxlen=1000,
            stochastic=stochastic, use_unk=use_unk)
        text = list()
        if stochastic:
            sample = [sample]
        for c in sample:
            text.append(' '.join([dec['word_idict'][w] for w in c[:-1]]))
        # sort beams by their NLL, return the best result
        lengths = numpy.array([len(s.split()) for s in text])
        if lengths[0] == 0: # in case the model only predicts <eos>
            lengths = lengths[1:]
            score = score[1:]
            text = text[1:]
        sidx = numpy.argmin(score)
        text = text[sidx]
        score = score[sidx]
        return text

    def tanh(self, x):
        """
        Tanh activation function
        """
        return tensor.tanh(x)

class Embedding(Base):
    """
    Joint image-sentence embedding space
    """
    def __init__(self):
        layers = {
            'ff': ('param_init_fflayer', 'fflayer'),
            'gru': ('param_init_gru', 'gru_layer')}

    def build_image_encoder(self, tparams, options):
        """
        Encoder only, for images
        """
        opt_ret = dict()
        trng = theano.sandbox.rng_mrg.MRG_RandomStreams(1234)
        im = theano.tensor.matrix('im', dtype='float32')
        images = self.get_layer('ff')[1](
            tparams, im, options, prefix='ff_image', activ='linear')
        images = self.l2norm(images)
        return trng, [im], images

    def build_sentence_encoder(self, tparams, options):
        """
        Encoder only, for sentences
        """
        opt_ret = dict()
        trng = theano.sandbox.rng_mrg.MRG_RandomStreams(1234)
        x = theano.tensor.matrix('x', dtype='int64')
        mask = theano.tensor.matrix('x_mask', dtype='float32')
        n_timesteps = x.shape[0]
        n_samples = x.shape[1]
        emb = tparams['Wemb'][x.flatten()].reshape(
            [n_timesteps, n_samples, options['dim_word']])
        proj = self.get_layer(options['encoder'])[1](
            tparams, emb, None, options, prefix='encoder', mask=mask)
        sents = proj[0][-1]
        sents = self.l2norm(sents)
        return trng, [x, mask], sents

    def encode_images(self, model, IM):
        """
        Encode images into the joint embedding space
        """
        images = model['f_ienc'](IM)
        return images

    def encode_sentences(self, model, X, verbose=False, batch_size=128):
        """
        Encode sentences into the joint embedding space
        """
        features = numpy.zeros((len(X), model['options']['dim']), dtype='float32')
        ds = collections.defaultdict(list)
        captions = [s.split() for s in X]
        for i,s in enumerate(captions):
            ds[len(s)].append(i)
        d = collections.defaultdict(lambda: 0)
        for w in model['worddict'].keys():
            d[w] = 1
        for k in ds.keys():
            if verbose:
                print(k)
            numbatches = len(ds[k]) / batch_size + 1
            for minibatch in range(numbatches):
                caps = ds[k][minibatch::numbatches]
                caption = [captions[c] for c in caps]
                seqs = list()
                for i, cc in enumerate(caption):
                    seqs.append(
                        [model['worddict'][w] if d[w] > 0 and 
                         model['worddict'][w] < model['options']['n_words'] else 1 for w in cc])
                x = numpy.zeros((k + 1, len(caption))).astype('int64')
                x_mask = numpy.zeros((k + 1, len(caption))).astype('float32')
                for idx, s in enumerate(seqs):
                    x[:k, idx] = s
                    x_mask[:k+1,idx] = 1.
                ff = model['f_senc'](x, x_mask)
                for ind, c in enumerate(caps):
                    features[c] = ff[ind]
        return features

    def gru_layer(
        self, 
        tparams,
        state_below,
        init_state,
        options,
        prefix='gru',
        mask=None,
        one_step=False,
        **kwargs):
        """
        Feedforward pass through GRU
        """
        nsteps = state_below.shape[0]
        if state_below.ndim == 3:
            n_samples = state_below.shape[1]
        else:
            n_samples = 1
        dim = tparams[self._p(prefix,'Ux')].shape[1]
        if init_state == None:
            init_state = theano.tensor.alloc(0.0, n_samples, dim)
        if mask == None:
            mask = theano.tensor.alloc(1., state_below.shape[0], 1)

        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:, :, n*dim:(n+1)*dim]
            return _x[:, n*dim:(n+1)*dim]

        state_below_ = theano.tensor.dot(
            state_below, tparams[self._p(prefix, 'W')]) + tparams[self._p(prefix, 'b')]
        state_belowx = theano.tensor.dot(
            state_below, tparams[self._p(prefix, 'Wx')]) + tparams[self._p(prefix, 'bx')]
        U = tparams[self._p(prefix, 'U')]
        Ux = tparams[self._p(prefix, 'Ux')]

        def _step_slice(m_, x_, xx_, h_, U, Ux):
            preact = theano.tensor.dot(h_, U)
            preact += x_
            r = theano.tensor.nnet.sigmoid(_slice(preact, 0, dim))
            u = theano.tensor.nnet.sigmoid(_slice(preact, 1, dim))
            preactx = theano.tensor.dot(h_, Ux)
            preactx = preactx * r
            preactx = preactx + xx_
            h = theano.tensor.tanh(preactx)
            h = u * h_ + (1. - u) * h
            h = m_[:,None] * h + (1. - m_)[:,None] * h_
            return h

        seqs = [mask, state_below_, state_belowx]
        _step = _step_slice
        if one_step:
            rval = _step(*(seqs + [
                init_state,
                tparams[self._p(prefix, 'U')],
                tparams[self._p(prefix, 'Ux')]]))
        else:
            rval, updates = theano.scan(
                _step, sequences=seqs, outputs_info=[init_state],
                non_sequences=[
                    tparams[self._p(prefix, 'U')],
                    tparams[self._p(prefix, 'Ux')]],
                name=self._p(prefix, '_layers'), n_steps=nsteps, profile=False, strict=True)
        rval = [rval]
        return rval

    def init_params(self, options):
        """
        Initialize all parameters
        """
        params = collections.OrderedDict()
        params['Wemb'] = self.norm_weight(options['n_words'], options['dim_word'])
        params = self.get_layer(options['encoder'])[0](
            options, params, prefix='encoder', nin=options['dim_word'], dim=options['dim'])
        params = self.get_layer('ff')[0](
            options, params, prefix='ff_image', nin=options['dim_image'], nout=options['dim'])
        return params

    def l2norm(self, X):
        """
        Compute L2 norm, row-wise
        """
        norm = theano.tensor.sqrt(theano.tensor.pow(X, 2).sum(1))
        X /= norm[:, None]
        return X

    def load_model(self, path_to_model):
        """
        Load all model components
        """
        with open('%s.dictionary.pkl'%path_to_model, 'rb') as f:
            worddict = cPickle.load(f)
        word_idict = dict()
        for kk, vv in worddict.iteritems():
            word_idict[vv] = kk
        word_idict[0] = '<eos>'
        word_idict[1] = 'UNK'
        with open('%s.pkl'%path_to_model, 'rb') as f:
            options = cPickle.load(f)
        params = self.init_params(options)
        params = self.load_params(path_to_model, params)
        tparams = self.init_tparams(params)
        trng = theano.sandbox.rng_mrg.MRG_RandomStreams(1234)
        trng, [x, x_mask], sentences = self.build_sentence_encoder(tparams, options)
        f_senc = theano.function([x, x_mask], sentences, name='f_senc')
        trng, [im], images = self.build_image_encoder(tparams, options)
        f_ienc = theano.function([im], images, name='f_ienc')
        model = {}
        model['options'] = options
        model['worddict'] = worddict
        model['word_idict'] = word_idict
        model['f_senc'] = f_senc
        model['f_ienc'] = f_ienc
        return model

    def param_init_fflayer(
        self,
        options,
        params,
        prefix='ff',
        nin=None,
        nout=None,
        ortho=True):
        """
        Feedforward layer

        Affine transformation + point-wise nonlinearity
        """
        if nin == None:
            nin = options['dim_proj']
        if nout == None:
            nout = options['dim_proj']
        params[self._p(prefix,'W')] = self.xavier_weight(nin, nout)
        params[self._p(prefix,'b')] = numpy.zeros((nout,)).astype('float32')
        return params

    def tanh(self, x):
        """
        Tanh activation function
        """
        return theano.tensor.tanh(x)

    def xavier_weight(self, nin, nout=None):
        """
        Xavier init
        """
        if nout == None:
            nout = nin
        r = numpy.sqrt(6.) / numpy.sqrt(nin + nout)
        W = numpy.random.rand(nin, nout) * 2 * r - r
        return W.astype('float32')

class Generate:
    """
    Story generation
    """
    def __init__(self):
        PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True

    def build_convnet(self, path_to_vgg):
        """
        Construct VGG-19 convnet
        """
        net = {}
        net['input'] = lasagne.layers.InputLayer((None, 3, 224, 224))
        net['conv1_1'] = lasagne.layers.corrmm.Conv2DMMLayer(net['input'], 64, 3, pad=1)
        net['conv1_2'] = lasagne.layers.corrmm.Conv2DMMLayer(net['conv1_1'], 64, 3, pad=1)
        net['pool1'] = lasagne.layers.MaxPool2DLayer(net['conv1_2'], 2)
        net['conv2_1'] = lasagne.layers.corrmm.Conv2DMMLayer(net['pool1'], 128, 3, pad=1)
        net['conv2_2'] = lasagne.layers.corrmm.Conv2DMMLayer(net['conv2_1'], 128, 3, pad=1)
        net['pool2'] = lasagne.layers.MaxPool2DLayer(net['conv2_2'], 2)
        net['conv3_1'] = lasagne.layers.corrmm.Conv2DMMLayer(net['pool2'], 256, 3, pad=1)
        net['conv3_2'] = lasagne.layers.corrmm.Conv2DMMLayer(net['conv3_1'], 256, 3, pad=1)
        net['conv3_3'] = lasagne.layers.corrmm.Conv2DMMLayer(net['conv3_2'], 256, 3, pad=1)
        net['conv3_4'] = lasagne.layers.corrmm.Conv2DMMLayer(net['conv3_3'], 256, 3, pad=1)
        net['pool3'] = lasagne.layers.MaxPool2DLayer(net['conv3_4'], 2)
        net['conv4_1'] = lasagne.layers.corrmm.Conv2DMMLayer(net['pool3'], 512, 3, pad=1)
        net['conv4_2'] = lasagne.layers.corrmm.Conv2DMMLayer(net['conv4_1'], 512, 3, pad=1)
        net['conv4_3'] = lasagne.layers.corrmm.Conv2DMMLayer(net['conv4_2'], 512, 3, pad=1)
        net['conv4_4'] = lasagne.layers.corrmm.Conv2DMMLayer(net['conv4_3'], 512, 3, pad=1)
        net['pool4'] = lasagne.layers.MaxPool2DLayer(net['conv4_4'], 2)
        net['conv5_1'] = lasagne.layers.corrmm.Conv2DMMLayer(net['pool4'], 512, 3, pad=1)
        net['conv5_2'] = lasagne.layers.corrmm.Conv2DMMLayer(net['conv5_1'], 512, 3, pad=1)
        net['conv5_3'] = lasagne.layers.corrmm.Conv2DMMLayer(net['conv5_2'], 512, 3, pad=1)
        net['conv5_4'] = lasagne.layers.corrmm.Conv2DMMLayer(net['conv5_3'], 512, 3, pad=1)
        net['pool5'] = lasagne.layers.MaxPool2DLayer(net['conv5_4'], 2)
        net['fc6'] = lasagne.layers.DenseLayer(net['pool5'], num_units=4096)
        net['fc7'] = lasagne.layers.DenseLayer(net['fc6'], num_units=4096)
        net['fc8'] = lasagne.layers.DenseLayer(net['fc7'], num_units=1000, nonlinearity=None)
        net['prob'] = lasagne.layers.NonlinearityLayer(
            net['fc8'], lasagne.nonlinearities.softmax)
        print('Loading parameters...')
        output_layer = net['prob']
        model = cPickle.load(open(path_to_vgg))
        lasagne.layers.set_all_param_values(output_layer, model['param values'])
        return net

    def compute_features(self, net, im):
        """
        Compute fc7 features for im
        """
        if config.FLAG_CPU_MODE:
            net.blobs['data'].reshape(* im.shape)
            net.blobs['data'].data[...] = im
            net.forward()
            fc7 = net.blobs['fc7'].data
        else:
            fc7 = numpy.array(lasagne.layers.get_output(
                net['fc7'], im, deterministic=True).eval())
        return fc7

    def load_image(self, file_name):
        """
        Load and preprocess an image
        """
        MEAN_VALUE = numpy.array([103.939, 116.779, 123.68]).reshape((3, 1, 1))
        image = PIL.Image.open(file_name)
        im = numpy.array(image)
        if len(im.shape) == 2:
            im = im[:, :, numpy.newaxis]
            im = numpy.repeat(im, 3, axis=2)
        h, w, _ = im.shape
        if h < w:
            im = skimage.transform.resize(im, (256, w*256/h), preserve_range=True)
        else:
            im = skimage.transform.resize(im, (h*256/w, 256), preserve_range=True)
        # central crop to 224x224
        h, w, _ = im.shape
        im = im[h // 2 - 112: h // 2 + 112, w // 2 - 112: w // 2 + 112]
        rawim = numpy.copy(im).astype('uint8')
        # shuffle axes to c01
        im = numpy.swapaxes(numpy.swapaxes(im, 1, 2), 0, 1)
        # convert to BGR
        im = im[::-1, :, :]
        im = im - MEAN_VALUE
        return rawim, lasagne.utils.floatX(im[numpy.newaxis])

    def load_all(self):
        """
        Load everything we need for generating
        """
        print(config.paths['decmodel'])
        print('Loading skip-thoughts...')
        stv = SkipThoughts.load_model(config.paths['skmodels'], config.paths['sktables'])
        print('Loading decoder...')
        dec = Decoder.load_model(config.paths['decmodel'], config.paths['dictionary'])
        print('Loading image-sentence embedding...')
        vse = Embedding.load_model(config.paths['vsemodel'])
        print('Loading and initializing ConvNet...')
        if config.FLAG_CPU_MODE:
            sys.path.insert(0, config.paths['pycaffe'])
            caffe.set_mode_cpu()
            net = caffe.Net(
                config.paths['vgg_proto_caffe'], config.paths['vgg_model_caffe'], caffe.TEST)
        else:
            net = self.build_convnet(config.paths['vgg'])
        print('Loading captions...')
        cap = list()
        with open(config.paths['captions'], 'rb') as f:
            for line in f:
                cap.append(line.strip())
        print('Embedding captions...')
        cvec = embedding.encode_sentences(vse, cap, verbose=False)
        print('Loading biases...')
        bneg = numpy.load(config.paths['negbias'])
        bpos = numpy.load(config.paths['posbias'])
        z = {}
        z['stv'] = stv
        z['dec'] = dec
        z['vse'] = vse
        z['net'] = net
        z['cap'] = cap
        z['cvec'] = cvec
        z['bneg'] = bneg
        z['bpos'] = bpos
        return z

    def story(self, z, image_loc, k=100, bw=50, lyric=False):
        """
        Generate a story for an image at location image_loc
        """
        rawim, im = self.load_image(image_loc)
        feats = self.compute_features(z['net'], im).flatten()
        feats /= scipy.linalg.norm(feats)
        feats = embedding.encode_images(z['vse'], feats[None,:])
        # compute the nearest neighbours
        scores = numpy.dot(feats, z['cvec'].T).flatten()
        sorted_args = numpy.argsort(scores)[::-1]
        sentences = [z['cap'][a] for a in sorted_args[:k]]
        print('NEAREST-CAPTIONS: ')
        for s in sentences[:5]:
            print(s)
        print('')
        svecs = skipthoughts.encode(z['stv'], sentences, verbose=False)
        shift = svecs.mean(0) - z['bneg'] + z['bpos']
        passage = decoder.run_sampler(z['dec'], shift, beam_width=bw)
        print('OUTPUT: ')
        if lyric:
            for line in passage.split(','):
                if line[0] != ' ':
                    print(line)
                else:
                    print(line[1:])
        else:
            print(passage)

class Search:
    """
    Code for sequence generation
    """
    def gen_sample(
        self,
        tparams,
        f_init,
        f_next,
        ctx,
        options,
        trng=None,
        k=1,
        maxlen=30,
        stochastic=True,
        argmax=False,
        use_unk=False):
        """
        Generate a sample, using either beam search or stochastic sampling
        """
        if k > 1:
            assert not stochastic, 'Beam search does not support stochastic sampling'
        sample, sample_score = list(), list()
        if stochastic:
            sample_score = 0
        live_k, dead_k = 1, 0
        hyp_samples = [[]] * live_k
        hyp_scores = numpy.zeros(live_k).astype('float32')
        hyp_states = list()
        next_state = f_init(ctx)
        next_w = -1 * numpy.ones((1,)).astype('int64')
        for ii in xrange(maxlen):
            inps = [next_w, next_state]
            ret = f_next(*inps)
            next_p, next_w, next_state = ret[0], ret[1], ret[2]
            if stochastic:
                if argmax:
                    nw = next_p[0].argmax()
                else:
                    nw = next_w[0]
                sample.append(nw)
                sample_score += next_p[0,nw]
                if nw == 0:
                    break
            else:
                cand_scores = hyp_scores[:,None] - numpy.log(next_p)
                cand_flat = cand_scores.flatten()
                if not use_unk:
                    voc_size = next_p.shape[1]
                    for xx in range(len(cand_flat) / voc_size):
                        cand_flat[voc_size * xx + 1] = 1e20
                ranks_flat = cand_flat.argsort()[:(k-dead_k)]
                voc_size = next_p.shape[1]
                trans_indices = ranks_flat / voc_size
                word_indices = ranks_flat % voc_size
                costs = cand_flat[ranks_flat]
                new_hyp_samples = list()
                new_hyp_scores = numpy.zeros(k-dead_k).astype('float32')
                new_hyp_states = list()
                for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                    new_hyp_samples.append(hyp_samples[ti]+[wi])
                    new_hyp_scores[idx] = copy.copy(costs[idx])
                    new_hyp_states.append(copy.copy(next_state[ti]))
                # check the finished samples
                new_live_k = 0
                hyp_samples, hyp_scores, hyp_states = [], [], []
                for idx in xrange(len(new_hyp_samples)):
                    if new_hyp_samples[idx][-1] == 0:
                        sample.append(new_hyp_samples[idx])
                        sample_score.append(new_hyp_scores[idx])
                        dead_k += 1
                    else:
                        new_live_k += 1
                        hyp_samples.append(new_hyp_samples[idx])
                        hyp_scores.append(new_hyp_scores[idx])
                        hyp_states.append(new_hyp_states[idx])
                hyp_scores = numpy.array(hyp_scores)
                live_k = new_live_k
                if new_live_k < 1:
                    break
                if dead_k >= k:
                    break
                next_w = numpy.array([w[-1] for w in hyp_samples])
                next_state = numpy.array(hyp_states)
        if not stochastic: # dump every remaining one
            if live_k > 0:
                for idx in xrange(live_k):
                    sample.append(hyp_samples[idx])
                    sample_score.append(hyp_scores[idx])
        return sample, sample_score

class SkipThoughts(Base):
    """
    Skip-thought vectors
    """
    def __init__(self):
        profile = False
        layers = {'gru': ('param_init_gru', 'gru_layer')}

    def build_encoder(self, tparams, options):
        """
        build an encoder, given pre-computed word embeddings
        """
        embedding = theano.tensor.tensor3('embedding', dtype='float32')
        x_mask = theano.tensor.matrix('x_mask', dtype='float32')
        proj = self.get_layer(options['encoder'])[1](
            tparams, embedding, options, prefix='encoder', mask=x_mask)
        ctx = proj[0][-1]
        return embedding, x_mask, ctx

    def build_encoder_bi(self, tparams, options):
        """
        build bidirectional encoder, given pre-computed word embeddings
        """
        embedding = theano.tensor.tensor3('embedding', dtype='float32')
        embeddingr = embedding[::-1]
        x_mask = theano.tensor.matrix('x_mask', dtype='float32')
        xr_mask = x_mask[::-1]
        proj = self.get_layer(options['encoder'])[1](
            tparams, embedding, options, prefix='encoder', mask=x_mask)
        projr = self.get_layer(options['encoder'])[1](
            tparams, embeddingr, options, prefix='encoder_r', mask=xr_mask)
        ctx = theano.tensor.concatenate([proj[0][-1], projr[0][-1]], axis=1)
        return embedding, x_mask, ctx

    def encode(self, model, X, use_norm=True, verbose=True, batch_size=128, use_eos=False):
        """
        Encode sentences in the list X. Each entry will return a vector
        """
        X = self.preprocess(X)
        # word dictionary and init
        d = collections.defaultdict(lambda : 0)
        for w in model['utable'].keys():
            d[w] = 1
        ufeatures = numpy.zeros((len(X), model['uoptions']['dim']), dtype='float32')
        bfeatures = numpy.zeros((len(X), 2 * model['boptions']['dim']), dtype='float32')
        # length dictionary
        ds = collections.defaultdict(list)
        captions = [s.split() for s in X]
        for i,s in enumerate(captions):
            ds[len(s)].append(i)
        # Get features. This encodes by length, in order to avoid wasting computation.
        for k in ds.keys():
            if verbose:
                print(k)
            numbatches = len(ds[k]) / batch_size + 1
            for minibatch in range(numbatches):
                caps = ds[k][minibatch::numbatches]
                if use_eos:
                    uembedding = numpy.zeros(
                        (k+1, len(caps), model['uoptions']['dim_word']), dtype='float32')
                    bembedding = numpy.zeros(
                        (k+1, len(caps), model['boptions']['dim_word']), dtype='float32')
                else:
                    uembedding = numpy.zeros(
                        (k, len(caps), model['uoptions']['dim_word']), dtype='float32')
                    bembedding = numpy.zeros(
                        (k, len(caps), model['boptions']['dim_word']), dtype='float32')
                for ind, c in enumerate(caps):
                    caption = captions[c]
                    for j in range(len(caption)):
                        if d[caption[j]] > 0:
                            uembedding[j,ind] = model['utable'][caption[j]]
                            bembedding[j,ind] = model['btable'][caption[j]]
                        else:
                            uembedding[j,ind] = model['utable']['UNK']
                            bembedding[j,ind] = model['btable']['UNK']
                    if use_eos:
                        uembedding[-1,ind] = model['utable']['<eos>']
                        bembedding[-1,ind] = model['btable']['<eos>']
                if use_eos:
                    uff = model['f_w2v'](
                        uembedding, numpy.ones((len(caption)+1,len(caps)), dtype='float32'))
                    bff = model['f_w2v2'](
                        bembedding, numpy.ones((len(caption)+1,len(caps)), dtype='float32'))
                else:
                    uff = model['f_w2v'](
                        uembedding, numpy.ones((len(caption),len(caps)), dtype='float32'))
                    bff = model['f_w2v2'](
                        bembedding, numpy.ones((len(caption),len(caps)), dtype='float32'))
                if use_norm:
                    for j in range(len(uff)):
                        uff[j] /= scipy.linalg.norm(uff[j])
                        bff[j] /= scipy.linalg.norm(bff[j])
                for ind, c in enumerate(caps):
                    ufeatures[c] = uff[ind]
                    bfeatures[c] = bff[ind]
        features = numpy.c_[ufeatures, bfeatures]
        return features

    def gru_layer(self, tparams, state_below, options, prefix='gru', mask=None, **kwargs):
        """
        Forward pass through GRU layer
        """
        nsteps = state_below.shape[0]
        if state_below.ndim == 3:
            n_samples = state_below.shape[1]
        else:
            n_samples = 1
        dim = tparams[self._p(prefix,'Ux')].shape[1]
        if mask == None:
            mask = theano.tensor.alloc(1., state_below.shape[0], 1)
        def _slice(_x, n, dim):
            if _x.ndim == 3:
                return _x[:,:, n * dim: (n+1) * dim]
            return _x[:, n * dim: (n+1) * dim]
        state_below_ = theano.tensor.dot(
            state_below, tparams[self._p(prefix, 'W')]) + tparams[self._p(prefix, 'b')]
        state_belowx = theano.tensor.dot(
            state_below, tparams[self._p(prefix, 'Wx')]) + tparams[self._p(prefix, 'bx')]
        U = tparams[self._p(prefix, 'U')]
        Ux = tparams[self._p(prefix, 'Ux')]

        def _step_slice(m_, x_, xx_, h_, U, Ux):
            preact = theano.tensor.dot(h_, U)
            preact += x_
            r = theano.tensor.nnet.sigmoid(_slice(preact, 0, dim))
            u = theano.tensor.nnet.sigmoid(_slice(preact, 1, dim))
            preactx = theano.tensor.dot(h_, Ux)
            preactx = preactx * r
            preactx = preactx + xx_
            h = tensor.tanh(preactx)
            h = u * h_ + (1. - u) * h
            h = m_[:,None] * h + (1. - m_)[:,None] * h_
            return h

        seqs = [mask, state_below_, state_belowx]
        _step = _step_slice
        rval, updates = theano.scan(
            _step, sequences=seqs, outputs_info = [tensor.alloc(0., n_samples, dim)],
            non_sequences = [tparams[self._p(prefix, 'U')], tparams[self._p(prefix, 'Ux')]],
            name=self._p(prefix, '_layers'), n_steps=nsteps, profile=profile, strict=True)
        rval = [rval]
        return rval

    def init_params(self, options):
        """
        initialize all parameters needed for the encoder
        """
        params = collections.OrderedDict()
        params['Wemb'] = self.norm_weight(options['n_words_src'], options['dim_word'])
        params = self.get_layer(options['encoder'])[0](
            options, params, prefix='encoder', nin=options['dim_word'], dim=options['dim'])
        return params

    def init_params_bi(self, options):
        """
        initialize all paramters needed for bidirectional encoder
        """
        params = collections.OrderedDict()
        params['Wemb'] = self.norm_weight(options['n_words_src'], options['dim_word'])
        params = self.get_layer(options['encoder'])[0](
            options, params, prefix='encoder', nin=options['dim_word'], dim=options['dim'])
        params = self.get_layer(options['encoder'])[0](
            options, params, prefix='encoder_r', nin=options['dim_word'], dim=options['dim'])
        return params

    def load_model(self, path_to_models, path_to_tables):
        """
        Load the model with saved tables
        """
        path_to_umodel = path_to_models + 'uni_skip.npz'
        path_to_bmodel = path_to_models + 'bi_skip.npz'
        # load model options
        with open('%s.pkl' % path_to_umodel, 'rb') as f:
            uoptions = cPickle.load(f)
        with open('%s.pkl' % path_to_bmodel, 'rb') as f:
            boptions = cPickle.load(f)
        # load parameters
        uparams = self.init_params(uoptions)
        uparams = self.load_params(path_to_umodel, uparams)
        utparams = self.init_tparams(uparams)
        bparams = self.init_params_bi(boptions)
        bparams = self.load_params(path_to_bmodel, bparams)
        btparams = self.init_tparams(bparams)
        # extractor functions
        embedding, x_mask, ctxw2v = build_encoder(utparams, uoptions)
        f_w2v = theano.function([embedding, x_mask], ctxw2v, name='f_w2v')
        embedding, x_mask, ctxw2v = self.build_encoder_bi(btparams, boptions)
        f_w2v2 = theano.function([embedding, x_mask], ctxw2v, name='f_w2v2')
        utable, btable = self.load_tables(path_to_tables)
        # store everything we need in a dictionary
        model = {}
        model['uoptions'] = uoptions
        model['boptions'] = boptions
        model['utable'] = utable
        model['btable'] = btable
        model['f_w2v'] = f_w2v
        model['f_w2v2'] = f_w2v2
        return model

    def load_tables(self, path_to_tables):
        """
        Load the tables
        """
        words = list()
        utable = numpy.load(path_to_tables + 'utable.npy')
        btable = numpy.load(path_to_tables + 'btable.npy')
        f = open(path_to_tables + 'dictionary.txt', 'rb')
        for line in f:
            words.append(line.decode('utf-8').strip())
        f.close()
        utable = collections.OrderedDict(zip(words, utable))
        btable = collections.OrderedDict(zip(words, btable))
        return utable, btable

    def preprocess(self, text):
        """
        Preprocess text for encoder
        """
        X = list()
        sent_detector = nltk.data.load('tokenizers/punkt/english.pkl')
        for t in text:
            sents = sent_detector.tokenize(t)
            result = ''
            for s in sents:
                tokens = nltk.tokenize.word_tokenize(s)
                result += ' ' + ' '.join(tokens)
            X.append(result)
        return X
