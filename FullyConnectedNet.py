from builtins import range
from builtins import object
import numpy as np

from layers import *
from layer_utils import *

class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function. This will also implement
    dropout and batch normalization as options. For a network with L layers,
    the architecture will be

    {affine - [batch norm] - relu - [dropout]} x (L - 1) - affine - softmax

    where batch normalization and dropout are optional, and the {...} block is
    repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=28*28, num_classes=10,
                 dropout=0, use_batchnorm=False, reg=0.0,
                 weight_scale=1e-2, dtype=np.float32, seed=None):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - dropout: Scalar between 0 and 1 giving dropout strength. If dropout=0 then
          the network should not use dropout at all.
        - use_batchnorm: Whether or not the network should use batch normalization.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A numpy datatype object; all computations will be performed using
          this datatype. float32 is faster but less accurate, so you should use
          float64 for numeric gradient checking.
        - seed: If not None, then pass this random seed to the dropout layers. This
          will make the dropout layers deteriminstic so we can gradient check the
          model.
        """
        self.use_batchnorm = use_batchnorm
        self.use_dropout = dropout > 0
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        ############################################################################
        # TODO: Initialize the parameters of the network, storing all values in    #
        # the self.params dictionary. Store weights and biases for the first layer #
        # in W1 and b1; for the second layer use W2 and b2, etc. Weights should be #
        # initialized from a normal distribution with standard deviation equal to  #
        # weight_scale and biases should be initialized to zero.                   #
        #                                                                          #
        # When using batch normalization, store scale and shift parameters for the #
        # first layer in gamma1 and beta1; for the second layer use gamma2 and     #
        # beta2, etc. Scale parameters should be initialized to one and shift      #
        # parameters should be initialized to zero.                                #
        ############################################################################
        
        self.params['W1'] = weight_scale * np.random.randn(input_dim, hidden_dims[0])
        for i in range(len(hidden_dims)-1):
            self.params['W'+str(i+2)] = weight_scale * np.random.randn(hidden_dims[i], hidden_dims[i+1])
            self.params['b'+str(i+1)] = np.zeros(hidden_dims[i])
            if self.use_batchnorm:
                self.params['gamma'+str(i+1)] = np.ones(hidden_dims[i])
                self.params['beta'+str(i+1)] = np.zeros(hidden_dims[i])
        self.params['W'+str(len(hidden_dims)+1)]= weight_scale * np.random.randn(hidden_dims[len(hidden_dims)-1],num_classes)
        self.params['b'+str(len(hidden_dims))] = np.zeros(hidden_dims[len(hidden_dims)-1])
        self.params['b'+str(len(hidden_dims)+1)] = np.zeros(num_classes)
        
        #print(self.params['W1'])
        #print(self.params['W2'])
        #print(self.params['W3'])
        """
        for k,v in self.params.items():
            print(k, v.shape)
        """
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # When using dropout we need to pass a dropout_param dictionary to each
        # dropout layer so that the layer knows the dropout probability and the mode
        # (train / test). You can pass the same dropout_param to each dropout layer.
        self.dropout_param = {}
        if self.use_dropout:
            self.dropout_param = {'mode': 'train', 'p': dropout}
            if seed is not None:
                self.dropout_param['seed'] = seed

        # With batch normalization we need to keep track of running means and
        # variances, so we need to pass a special bn_param object to each batch
        # normalization layer. You should pass self.bn_params[0] to the forward pass
        # of the first batch normalization layer, self.bn_params[1] to the forward
        # pass of the second batch normalization layer, etc.
        self.bn_params = []
        if self.use_batchnorm:
            self.bn_params = [{'mode': 'train'} for i in range(self.num_layers - 1)]

        # Cast all parameters to the correct datatype
        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.

        Input / output: Same as TwoLayerNet above.
        """
        X = X.astype(self.dtype)
        mode = 'test' if y is None else 'train'
        num_layers = self.num_layers

        # Set train/test mode for batchnorm params and dropout param since they
        # behave differently during training and testing.
        if self.use_dropout:
            self.dropout_param['mode'] = mode
        if self.use_batchnorm:
            for bn_param in self.bn_params:
                bn_param['mode'] = mode

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the fully-connected net, computing  #
        # the class scores for X and storing them in the scores variable.          #
        #                                                                          #
        # When using dropout, you'll need to pass self.dropout_param to each       #
        # dropout forward pass.                                                    #
        #                                                                          #
        # When using batch normalization, you'll need to pass self.bn_params[0] to #
        # the forward pass for the first batch normalization layer, pass           #
        # self.bn_params[1] to the forward pass for the second batch normalization #
        # layer, etc.                                                              #
        ############################################################################
        cache = [None]*(num_layers+1)
        out = X
        for i in range(1,num_layers):
            cache[i] = {}
            W = self.params['W'+str(i)]
            b = self.params['b'+str(i)]
            #out, cache[i]['af'] = affine_forward(out,W,b)#affine
            #out, cache[i]['relu'] = relu_forward(out)#relu
            out, cache[i]['af-relu'] = affine_relu_forward(out,W,b)
            if self.use_batchnorm:
                gamma, beta = self.params['gamma'+str(i)], self.params['beta'+str(i)]
                bn_param = self.bn_params[i-1]
                out, cache[i]['bn'] = batchnorm_forward(out,gamma,beta,bn_param)#bn_normalization
            #out, cache[i]['relu'] = relu_forward(out)#relu
            if self.use_dropout:
                dropout_param = self.dropout_prama
                out, cache[i]['dp'] = dropout_forward(out, dropout_param)
        cache[num_layers] = {}
        scores, cache[num_layers]['af'] = affine_forward(out, self.params['W'+str(num_layers)], self.params['b'+str(num_layers)])
          
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        reg = self.reg
        ############################################################################
        # TODO: Implement the backward pass for the fully-connected net. Store the #
        # loss in the loss variable and gradients in the grads dictionary. Compute #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # When using batch normalization, you don't need to regularize the scale   #
        # and shift parameters.                                                    #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        loss, dout = softmax_loss(scores, y)
        R = 0
        for i in range(1,num_layers):
            R += np.sum(self.params['W'+str(i)]**2)                                            
        loss += reg * R/2
        dout, dW, db = affine_backward(dout,cache[num_layers]['af'])
        grads['W'+str(num_layers)] = dW + reg*self.params['W'+str(num_layers)]
        grads['b'+str(num_layers)] = db
        for i in range(1,num_layers):
            if self.use_dropout:
                dout = dropout_backward(dout, cache[num_layers-i]['dp'])
            #dout = relu_backward(dout, cache[num_layers-1-1-i]['relu'])                                     
            if self.use_batchnorm: 
                dout, dgamma, dbeta = batchnorm_backward(dout, cache[num_layers-i]['bn'])
                grads['gamma'+str(num_layers-1-i)] = dgamma
                grads['beta'+str(num_layers-1-i)] = dbeta
            #dout = relu_backward(dout, cache[num_layers-1-1-i]['relu'])
            #dout, dW, db = affine_backward(dout, cache[num_layers-1-1-i]['af'])
            dout, dW, db = affine_relu_backward(dout,cache[num_layers-i]['af-relu']) 
            grads['W'+str(num_layers-i)] = dW + reg*self.params['W'+str(num_layers-i)]
            grads['b'+str(num_layers-i)] = db
            
        #print(self.params['W1'])
        #print(self.params['W2'])
        #print(self.params['W3']) 
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
