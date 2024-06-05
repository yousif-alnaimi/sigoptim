import numba
from numba import jit,njit
import numpy as np
import copy

@njit([numba.int32(numba.int32,numba.int32)])
def length_of_signature(dim,level):
    """Length of signature of paths with `dim` channels at level `level`."""
    return (dim**level-1)//(dim-1)

@njit([numba.int32(numba.int32[:],numba.int32)])
def convert_indices2_position_in_signature(indices,dim):
    """ Given a word [a_1,a_2,...,a_n], get its position in a 1-d array returned by 
        common signature computation functions(like iisignautre[1] and signatory[2]).
        [1] https://pypi.org/project/iisignature/
        [2] https://pypi.org/project/signatory/
    """
    if len(indices)==0:
        return 0
    n = len(indices)
    index = 0
    # position at the level the indices lie in
    for i in range(n):
        index += indices[i]*dim**(n-i-1)
    level = len(indices)
    # add the number of elements in previous levels
    index += length_of_signature(dim,level)
    return index

"""Implementation of Shuffle and Half Shuffle Products. Could be replaced by other implementations."""

class Tree:
    def __init__(self, data):
        self.children = []
        self.data = data
        self.list = [[],[]]
        self.number = 2

def _get_paths(t, paths=None, current_path=None):
    if paths is None:
        paths = []
    if current_path is None:
        current_path = []

    current_path.append(t.data)
    if len(t.children) == 0:
        paths.append(current_path)
    else:
        for child in t.children:
            _get_paths(child, paths, list(current_path))
    return paths

def half_shuffle(x):
    """ Input: an iterable objects of lists l1,l2,..,ln
        Return: l1 \prec (l2 \prec (ln-1 \prec ln))))))"""
    for i in range(len(x)):
        x[i] = x[i][::-1]
    x0 = x[0].pop()
    root = Tree(x0)
    root.list = x
    # 
    stack = [root]
    while stack:
        current_node = stack.pop()
        for n in range(current_node.number):
            if current_node.list[n]:
                node_list = copy.deepcopy(current_node.list)
                node = Tree(node_list[n].pop())
                node.number = max([n+2,current_node.number])
                node.number = min([node.number,len(current_node.list)])
                node.list = node_list
                current_node.children.append(node)
        for node in current_node.children:
            stack.append(node)

    paths = _get_paths(root)

    return paths
    
def shuffle(x):
    """ Input: an iterable objects of lists l1,l2,..,ln
        Return l1*l2*...*ln
    """
    for i in range(len(x)):
        x[i] = x[i][::-1]
    paths = []
    for i in range(len(x)):
        x_c = copy.deepcopy(x)
        x0 = x_c[i].pop()
        root = Tree(x0)
        root.list = x_c

        stack = [root]
        while stack:
            current_node = stack.pop()
            for n in range(len(x)):
                if current_node.list[n]:
                    node_list = copy.deepcopy(current_node.list)
                    node = Tree(node_list[n].pop())
                    node.list = node_list
                    current_node.children.append(node)
            for node in current_node.children:
                stack.append(node)

        paths += _get_paths(root)

    return paths

""" Generate cartesian product of a numpy array with repeats.
    The code is from Hadrien Titeux in stackoverflow 
    https://stackoverflow.com/questions/57128975/cartesian-product-in-numba. 
    It can be replaced by itertools.poroduct"""

@njit(numba.int32[:,:](numba.int32[:]))
def cproduct_idx(sizes: np.ndarray):
    """Generates ids tuples for a cartesian product"""
    assert len(sizes) >= 2
    tuples_count  = np.prod(sizes)
    tuples = np.zeros((tuples_count, len(sizes)), dtype=np.int32)
    tuple_idx = 0
    # stores the current combination
    current_tuple = np.zeros(len(sizes))
    while tuple_idx < tuples_count:
        tuples[tuple_idx] = current_tuple
        current_tuple[0] += 1
        # using a condition here instead of including this in the inner loop
        # to gain a bit of speed: this is going to be tested each iteration,
        # and starting a loop to have it end right away is a bit silly
        if current_tuple[0] == sizes[0]:
            # the reset to 0 and subsequent increment amount to carrying
            # the number to the higher "power"
            current_tuple[0] = 0
            current_tuple[1] += 1
            for i in range(1, len(sizes) - 1):
                if current_tuple[i] == sizes[i]:
                    # same as before, but in a loop, since this is going
                    # to get run less often
                    current_tuple[i + 1] += 1
                    current_tuple[i] = 0
                else:
                    break
        tuple_idx += 1
    return tuples

@njit
def cartesian_product(*arrays):
    sizes = [len(a) for a in arrays]
    sizes = np.asarray(sizes, dtype=np.int8)
    tuples_count  = np.prod(sizes)
    array_ids = cproduct_idx(sizes)
    tuples = np.zeros((tuples_count, len(sizes)))
    for i in range(len(arrays)):
        tuples[:, i] = arrays[i][array_ids[:, i]]
    return tuples

@njit
def cartesian_product_repeat(array, repeat):
    sizes = [len(array) for _ in range(repeat)]
    sizes = np.asarray(sizes, dtype=np.int32)
    tuples_count  = np.prod(sizes)
    array_ids = cproduct_idx(sizes)
    tuples = np.zeros((tuples_count, len(sizes)),dtype=np.int32)
    for i in range(repeat):
        tuples[:, i] = array[array_ids[:, i]]
    return tuples

""" Here defines the used word operations: shuffle product, and first cancatenate a word representing 
    integration along one of the member path of the process, and then take shuffle product with another
    integrated paths."""

# shuffle product
def word_shuffle_product(level1,level2):
    """ Return a dictionary with '{i}{j}' as the keyword, where i, j are the length
        of the two word-operands, and is from 0 to `level1`\`level2` respectively. 
    """
    word_shuffle_dict = numba.typed.Dict.empty(key_type=numba.types.unicode_type,value_type=numba.types.int32[:,:])
    for i in range(1,level1):
        for j in range(1,level2):
            word_shuffle_dict[f'{i}{j}'] = np.array(shuffle([[1 for _ in range(i)],[2 for _ in range(j)]]),dtype=np.int32)
    for i in range(level1):
        word_shuffle_dict[f'{i}0'] = np.ones((1,i),dtype=np.int32)
    for i in range(1,level2):
        word_shuffle_dict[f'0{i}'] = np.ones((1,i),dtype=np.int32)
    return word_shuffle_dict

# integrate(concatenate a letter) and shuffle
def word_concatenate_shuffle(level1,level2):
    """ Return a dictionary with '{i}{j}' as the keyword, where i, j are the length
        of the two word-operands, and is from 0 to `level1`\`level2` respectively.
        The value of 'ij' should be the output of 1...1(i times)[-1] \shuffle 2...2(j times)[-2].
    """
    word_concatenate_shuffle_dict = numba.typed.Dict.empty(key_type=numba.types.unicode_type,value_type=numba.types.int32[:,:])
    for i in range(level1):
        for j in range(level2):
            # shuffle product between 1...1[-1] and 2...2[-2] 
            word_concatenate_shuffle_dict[f'{i}{j}'] = np.array(shuffle([[1 for _ in range(i)]+[-1],[2 for _ in range(j)]+[-2]]),dtype=np.int32)
    return word_concatenate_shuffle_dict

@jit(nopython=True)
#@vectorize([numba.float64(numba.int32[:,:],numba.float64[:],numba.int32[:],numba.int32,numba.int32)], target='parallel')
def apply_bioperation_to_word(x,signature,word_operation,dim,level,*o):
    """Given a word, return the position in the signature according to a pre-computed abstract word operation,
        I didn't see the way to generalize it, so user may require to define it every time for different operations.
        (like this is in fact defined for `word_concatenate_shuffle`)
        But it may be generalized in some way, although it is not necessary to do that(could be taken place by multiple inputs operation
        The generalization to multiple inputs is straightfoward!

        Arguments:
        x: an array of shape (level[0]+level[1],). Concatenated by word1 and word2
        signature: an array of shape (N,) N=(dim^m-1)(dim-1) the expected signature
        word_operation: an array obtained from functions like `word_concatenate_shuffle`[level1,level2].
            It contains  1's,2's,.. and other stuff (like -1's,-2's,...)
        dim: int dimension of the paths
        level: a tuple (level1,level2), the level of two words
        """
    result = 0
    # get the position of the word in signature
    i1 = convert_indices2_position_in_signature(x[:level[0]],dim)
    i2 = convert_indices2_position_in_signature(x[level[0]:],dim)

    for k in range(len(word_operation)):
        xx = word_operation[k]
        
        indices = np.zeros(len(xx),dtype=np.int32)
        indices[np.where(xx==1)] = x[:level[0]] # replace 1 by letters in w1
        indices[np.where(xx==2)] = x[level[0]:] # replace 2 by letters in w2
        indices[np.where(xx==-1)] = o[0] # replace -1 by m
        indices[np.where(xx==-2)] = o[1] # replace -2 by n

        index = convert_indices2_position_in_signature(indices,dim)
        #print(convert_indices2_position_in_signature(indices,dim))
        result += signature[index]
    return ((i1,i2),result)

@njit
def concatenate_a_letter_to_words(x,signature,dim,m):
    """Given a word, return the position of the word in signature and the corresponding coefficients"""
    i = convert_indices2_position_in_signature(x,dim) # get the position of the word in signature
    indices = np.concatenate((x,np.array([m],dtype=np.int32))) # get the indices

    # read the elements corresponding to the indices in the signature
    result = signature[convert_indices2_position_in_signature(indices,dim)] 
    return (i,result)

@jit(nopython=True,parallel=True)
def squared_integration_functional(signature,word_operation_dict,dim,level,m,n):
    """ Return the coefficeints matrix of a^m_p(w)a^n_p(w) for w in W^d_m(A), in 
        (l_m \prec m) * (l_n \prec n) where  W^d_m(A) is the set of words of 
        dimension d up to length m. 
        
        Arguments:
        -----------------------------
        word_operation_dict: dict   a dictionary containing expression of w_i, w_j with keywords 'ij'
        m,n: int    the specific channel 
        """
    sig_len = length_of_signature(dim,level) # length of signature at (level, dim)
    weights = np.zeros((sig_len,sig_len)) # (l_m < m) * (l_n < n) = sum weights_ij (a^m_(w_1) a^n_(w_2))
    for i in range(level):
        for j in range(level):
            if i+j ==0:
                # a^m_0 a^n_0 (m*n)
                ii1 = convert_indices2_position_in_signature(np.array([m,n],dtype=np.int32),dim) 
                ii2 = convert_indices2_position_in_signature(np.array([n,m],dtype=np.int32),dim)
                weights[0,0] = signature[ii1]+signature[ii2]
                continue
            elif i+j == 1:
                generator = np.array([p for p in range(dim)],dtype=np.int32).reshape(dim,1)
            else:
                generator = cartesian_product_repeat(np.arange(dim),i+j)
            for k in range(len(generator)):
                x = generator[k]
                # a^m_x1 a^n_x2 (x1 * x2), ii = (p(x1),p(x2), es = sum S()_(x1 * x2)
                ii, es = apply_bioperation_to_word(x,signature,word_operation_dict[f'{i}{j}'],dim,(i,j),m,n)

                weights[ii[0],ii[1]] = es
    return weights

def get_sig_variance(signature,word_operation_dict,dim,level,hoff=False):
    """ Get the coefficeints of total variance by block maztrixs (X2_ij)_{1<= i,j <= d}
        Hence the total square of the return is a.T@X2@a, where a_p(w) is the collection 
        of coefficeints of w."""
    sig_len = length_of_signature(dim,level)
    weights = np.zeros((sig_len*dim,sig_len*dim))
    if hoff:
        for m in range(dim, 2*dim):
            for n in range(dim, 2*dim):
                weights[m*sig_len:(m+1)*sig_len,n*sig_len:(n+1)*sig_len] = squared_integration_functional(signature,word_operation_dict,dim,level,m,n)
        return weights
    else:
        for m in range(dim):
            for n in range(dim):
                weights[m*sig_len:(m+1)*sig_len,n*sig_len:(n+1)*sig_len] = squared_integration_functional(signature,word_operation_dict,dim,level,m,n)
        return weights
        

@jit(nopython=True,parallel=True)
def integration_functional_coeff(signature,dim,level,m):
    """ Return the coefficeints vectors R_m of a^m_p(w) for w in W^d_m(A) in expected returns. 
        Hence E[l_m(S(X))] = R_m @ a, where a_p(w) is the collection of coefficeints of w"""
    coeff = np.zeros(length_of_signature(dim,level))

    coeff[0] = signature[m+1]   # when j=0, the coefficient of a_0 is ES(x)^m

    for j in range(1,level):
        if j == 1:
            generator = np.array([p for p in range(dim)],dtype=np.int32).reshape(dim,1)
        else:
            generator = cartesian_product_repeat(np.arange(dim),j)
        for k in range(len(generator)):
            # a^m_w w \prec m. Hence the coefficents is ES(X)^wm
            i,es = concatenate_a_letter_to_words(generator[k],signature,dim,m)
            coeff[i] += es
    return coeff

def get_weights_sum_coeff(signature,dim,level):
    """ Get coefficeints of a^m_w in the sum of weights of all assets as WS.
        sum w_i = WS @ a, where a^i_p(w) is the collection of coefficeints of w
        on the i'th channel"""
    sig_len = length_of_signature(dim,level)
    truncated_signature = signature[:sig_len]
    coeff =  np.tile(truncated_signature,dim)
    return coeff

@njit
def get_weights_coeff(signature,m,dim,level):
    """ Get coefficeints of a^m_w in the sum of weights of all assets as WS.
        w_i = W_i @ a, where a^i_p(w) is the collection of coefficeints of w
        on the i'th channel"""
    sig_len = length_of_signature(dim,level)
    coeff = np.zeros(sig_len*dim)
    coeff[m*sig_len:(m+1)*sig_len] = signature[:sig_len]
    return coeff


def HoffLeadLag(paths: np.ndarray, time_aug: bool=False) -> np.ndarray:
    """
    Performs Hoff lead-lag transformation on a bank of paths. Sends N x l x d to N x 4l + 1 x 2d.

    :param paths:   Bank to have Hoff lead-lag applied to
    :time_aug:      Whether the paths are time augmented
    :return:        Hoff lead-lag transformed paths
    """
    if time_aug:
        paths = paths[:, :, 1:]


    _r = np.repeat(paths, 4, axis=1)
    _cpath = np.concatenate([_r[:, :-5], _r[:, 5:]], axis=2)
    _start = np.expand_dims(np.c_[_r[:, 0], _r[:, 0]], 1)
    hoff_paths = np.concatenate([_start, _cpath], axis=1)

    time_steps = hoff_paths.shape[1]
    n_samples = hoff_paths.shape[0]

    time_inds = np.expand_dims(np.stack([np.linspace(0,1,time_steps) for _ in range(n_samples)], axis=0), axis=2)
    return np.concatenate((time_inds, hoff_paths), axis=2)


def subtract_first_row(paths: np.ndarray) -> np.ndarray:
    """
    Subtract the first row of the paths from all rows in the paths.

    :param paths:   Paths to have the first row subtracted from
    :return:        Paths with the first row subtracted
    """
    _, l, _ = paths.shape

    return paths - np.tile(np.expand_dims(paths[:, 0, :], 1), (1, l, 1))


# print(np.array([[[1,2,3],[4,5,6],[7,8,9]],[[1,2,3],[4,5,6],[7,8,9]]]))
# print(subtract_first_row(np.array([[[1,2,3, 4],[4,5,6, 4],[7,8,9, 4]],[[1,2,3, 5],[4,5,6,5],[7,8,9,5]]])))