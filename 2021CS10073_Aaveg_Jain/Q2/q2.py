# %%
# %pip install cvxopt

# %%
import cvxopt 
from cvxopt import matrix
from cvxopt import solvers

import matplotlib.pyplot as plt
import math

# %% [markdown]
# 

# %%
# !unzip "./svm/drive/MyDrive/svm.zip"

# %%
# %pip install opencv-python
import cv2 as cv2 
import numpy as np
import os

# %%
def resize(img_name, new_size):
  img = cv2.imread(img_name)
  img = img.astype(np.float32)
  img = cv2.resize(img,  new_size)
  norm_img = img/255.0
  norm_img = norm_img.flatten()

  return norm_img

# %%
# design matrix
folder_name = "./svm/train/3"

X = tuple()
new_shape = (16, 16)
cnt = 2

for filename in os.listdir(folder_name):
      file = os.path.join(folder_name, filename)
      Y = resize(file, new_shape)
      X = X + ( Y[None , ...], )


folder_name = "./svm/train/4"
for filename in os.listdir(folder_name):
      file = os.path.join(folder_name, filename)
      Y = resize(file, new_shape)
      X = X + ( Y[None , ...], )

X = np.vstack(X)


# %%
# labels
Y = np.array([], dtype = int)
for i in range( int(X.shape[0]/2) ):
  Y = np.append(Y, -1)
for i in range(int(X.shape[0]/2)):
  Y = np.append(Y, 1)



# %%
# modelling the problem as a cvx opt problem

class SVM:
  def __init__(self, X, Y, C):
    self.X = X
    self.Y = Y
    self.C = C
    self.m = Y.shape[0]
    # print("here")
    # self.P = matrix(self.get_P()) for general kernel case 
    self.linear_kernel_matrix()
    self.P = matrix(self.kernel_matrix) 
    self.q = matrix(np.full( (self.m, ), -1 ))

    temp = np.full( (self.m, ) , 1)
    temp2 = np.full( (self.m, ) , -1)
    self.G = matrix(np.vstack( ( np.diag(temp2), np.diag(temp))  ) )

    temp = np.full( (self.m, ), C)
    self.h = matrix(np.hstack( ( np.zeros( (self.m, ) ), temp) ) )

    self.A = matrix(np.vstack( (self.Y, )))
    self.b = matrix(np.zeros( (1,) ))

    self.P = matrix( np.array(self.P), tc='d')
    self.q = matrix( np.array(self.q), tc='d')
    self.G = matrix( np.array(self.G), tc='d')
    self.h = matrix( np.array(self.h), tc='d')
    self.A = matrix( np.array(self.A), tc='d')
    self.b = matrix( np.array(self.b), tc='d')
    self.no_of_classes = 2

    self.soln = None


  def kernel(self, x, y):
    return (x.T) @ y # override for different kernels


  def gaussian_kernel_matrix(self): # custom function for gaussian kernel
    
    pairwise_sq_dists = np.sum(self.X**2 , axis=1, keepdims=True) - 2 * np.dot(self.X, self.X.T) + np.sum(self.X**2, axis=1, keepdims=True).T
    self.kernel_matrix = np.exp(- self.gamma * pairwise_sq_dists)
    # print(np.outer(self.Y, self.Y).shape) 
    self.kernel_matrix = np.outer(self.Y, self.Y) * self.kernel_matrix
    return 
  
  def linear_kernel_matrix(self): # custom function for linear kernel 
    temp = np.outer(self.Y, self.Y) 
    self.kernel_matrix = temp * (self.X @ self.X.T) 

  def get_P(self):
    new_shape = (self.m , self.m)
    temp = np.zeros(new_shape)
    for i in range(self.m):
      for j in range(self.m):
        temp[i, j] = self.Y[i] * self.Y[j] * self.kernel(self.X[i, :], self.X[j, :])
    return temp

  def solve(self):
    self.soln = solvers.qp(self.P, self.q, self.G, self.h, self.A, self.b)
    return self.soln

  def get_support_vectors(self):
    alphas = np.array(self.soln["x"])
    alphas = np.array( [ alpha[0] for alpha in alphas] )
    # print(alphas)
    alphas = [[alphas[i],  int(i) ] for i in range(alphas.shape[0])]
    alphas = np.array(alphas)
    self.og_alphas = alphas
    sorted_indices = np.argsort(self.og_alphas[:, 1])
    self.og_alphas = self.og_alphas[sorted_indices]
    shape = alphas.shape
    m = shape[0]

    self.init_size = alphas.shape[0]
    support_vectors = np.array([ (alpha[0], alpha[1]) for alpha in alphas if (alpha[0] > 1e-05)] )
    sorted_indices = np.argsort(support_vectors[:, 0])
    self.support_vectors = support_vectors[sorted_indices]
    self.no_of_support_vectors = support_vectors.shape[0]
    print(self.init_size, self.no_of_support_vectors)
    return (self.og_alphas, self.support_vectors)

  def inner_product(self, x): # takes inner product of a input attribute' feature vector with weight vector
    # temp = np.array([self.kernel(self.X[i], x) for i in range(self.m)]) 
    temp = self.X @ x 
    return np.sum(self.og_alphas[:,0] * self.Y * temp)
  
  def w_norm(self): # applicable for all kernels
    temp = self.og_alphas[:, 0] 
    return math.sqrt(temp.T @ self.P @ temp) 
  
  def get_weight(self): # only valid for linear kernel, not gaussian kernel
    weight = np.zeros( (self.Y.shape[0], ) )
    temp = self.Y * (self.og_alphas[:,0])
    self.weight = ( (self.X).T @ temp)
    # weight = np.sum ( temp2,axis = 0)
    print(weight.shape)
    return self.weight

  def get_bias(self):
      # m = (self.og_alphas).shape[0]
      all_bias = []
      for i in range(self.m):
        alpha = self.og_alphas[i,0]
        if (alpha < 1e-05 or (self.C - alpha) < 1e-05): continue
        # temp = ((self.weight.T) @ self.X[i])
        temp = self.inner_product(self.X[i])
        bias = self.Y[i] - temp
        all_bias.append(bias)
      all_bias = np.array(all_bias)
      self.bias = np.sum(all_bias)/all_bias.shape[0]
      return self.bias

  def predict(self, example):
    temp = self.inner_product(example) + self.bias
    temp /= self.w_norm() 
    # print(temp)
    if (temp > 0) : predict = 1
    else: predict = -1
    return (predict, temp)

  def get_confusion_matrix(self, test_eg, test_ans) :
    size = test_eg.shape[0]
    confusion_matrix = np.zeros( (self.no_of_classes, self.no_of_classes))
    correct , incorrect = 0,0
    for k in range(size):
      (prediction, score) = self.predict(test_eg[k])
      if (test_ans[k] == prediction) : correct += 1
      else : incorrect +=  1
      if (prediction == -1) : prediction = 0
      copy = 1
      if (test_ans[k] == -1) : copy = 0
      confusion_matrix[copy][prediction] += 1


    return (confusion_matrix, (correct / (correct + incorrect)))




# %%
linear_kernel = SVM(X, Y , 1.0)

# %%
soln = linear_kernel.solve()

# %%

og_alphas, support_vectors = linear_kernel.get_support_vectors()
print(support_vectors)


# %%
# weight = get_weight(og_alphas, X, Y)
weight = linear_kernel.get_weight()
print(weight.shape)

# %%
# bias = get_bias(og_alphas, X, Y, weight, 1.0)
bias = linear_kernel.get_bias()
print(bias)
# bias =

# %% [markdown]
# Bias is 0.725

# %% [markdown]
# We get 2903 support vectors out of 4760 training examples. We take a margin of 1e0-5 for values of alpha.
# The percentage of suport vectors is 60.99 %

# %%
# validation matrix
folder_name = "./svm/val/3"

new_shape = (16, 16)

validation_eg = []
answers = []
correct , incorrect = 0,0
for filename in os.listdir(folder_name):
      answers.append(-1)
      file = os.path.join(folder_name, filename)
      temp = resize(file, new_shape)
      validation_eg.append(temp)

folder_name = "./svm/val/4"
for filename in os.listdir(folder_name):
      answers.append(1)
      file = os.path.join(folder_name, filename)
      temp = resize(file, new_shape)
      validation_eg.append(temp)

validation_eg = np.array(validation_eg)
answers = np.array(answers)

# X = np.vstack(X)



# %%
( confusion_matrix, accuracy) = linear_kernel.get_confusion_matrix(validation_eg, answers)
print(f"accuracy is {accuracy}")

# %% [markdown]
# We get validation accuracy as 72.75 %

# %%
class Gaussian_Kernel(SVM):
   def __init__(self, X, Y, C, gamma):
      self.X = X
      self.Y = Y
      self.C = C
      self.m = Y.shape[0]
      self.gamma = gamma
      # print("here in gaussian")
      self.gaussian_kernel_matrix()
      # print(self.kernel_matrix) 
      self.P = matrix(self.kernel_matrix)
      self.q = matrix(np.full( (self.m, ), -1 ))

      temp = np.full( (self.m, ) , 1)
      temp2 = np.full( (self.m, ) , -1)
      self.G = matrix(np.vstack( ( np.diag(temp2), np.diag(temp))  ) )

      temp = np.full( (self.m, ), C)
      self.h = matrix(np.hstack( ( np.zeros( (self.m, ) ), temp) ) )

      self.A = matrix(np.vstack( (self.Y, )))
      self.b = matrix(np.zeros( (1,) ))

      self.P = matrix( np.array(self.P), tc='d')
      self.q = matrix( np.array(self.q), tc='d')
      self.G = matrix( np.array(self.G), tc='d')
      self.h = matrix( np.array(self.h), tc='d')
      self.A = matrix( np.array(self.A), tc='d')
      self.b = matrix( np.array(self.b), tc='d')

      self.soln = None
      self.no_of_classes = 2


   def kernel(self, x, y): # not used, only for consistency for using get_P()
    #  temp = x - y
     norm_sq = (x - y).T @ (x - y)
     return math.exp(-self.gamma * norm_sq)

# %%
C = 1.0
gamma = 0.001
gaussian_kernel = Gaussian_Kernel(X, Y, C, gamma)

# %%
soln2 = gaussian_kernel.solve()

# %%
gaussian_kernel.get_support_vectors()
gaussian_kernel.get_bias()

# %%
(confusion_matrix, accuracy) = gaussian_kernel.get_confusion_matrix(validation_eg, answers)
print(f"accuracy is {accuracy}")

# %%
# plotting images
def plot(svm, svm_name, flag = True ):
  top_6 = svm.support_vectors[-6:]

  # print(top_6)
  top_6 = np.array([ (svm.X[ int(sv[1]) ]) * 255.0 for sv in top_6])
  top_6 = np.array([ img.reshape((16,16,3)) for img in top_6])
  cnt = 0
  for img in top_6:

    cnt += 1
    cv2.imwrite(f"img_{cnt}_{svm_name}.png", img)

  if (flag):
    resized_weight = weight * 255.0
    resized_weight = resized_weight.reshape( (16, 16, 3) )
    cv2.imwrite(f"weight_{svm_name}.png", resized_weight)

def get_common_support_vectors(svm1, svm2):
    v1 = set( svm1.support_vectors[:, 1] )
    v2 = set( svm2.support_vectors[:, 1]   )
    print(f"lens are {len(v1)} and {len(v2)}")
    common = v1 & v2 # intersection of two sets
    return len(common)



# %%
plot(linear_kernel, "Linear")
plot(gaussian_kernel, "Gaussian", False)
print(get_common_support_vectors(linear_kernel, gaussian_kernel))


# %% [markdown]
# Linear kernel had 2903 (60.99 %) support vectors, and Gaussian has 3453 (72.54 %) support vectors. The common support vectors are 2698 (56.68 %) in count.

# %% [markdown]
# Validation accuracy for Gaussian is 77.75 % (compared to 72.75 % for linear kernel, an increase of 5 %) .

# %%
# !pip install scikit
from sklearn import svm

# %%
sk_linear_kernel = svm.SVC(C = 1.0, kernel = "linear")
sk_linear_kernel.fit(X, Y)

# %%
sk_gaussian_kernel = svm.SVC(C = 1.0, kernel = "rbf", gamma = 0.001)
sk_gaussian_kernel.fit(X, Y)

# %%
predictions = sk_linear_kernel.predict(validation_eg)
correct , incorrect = 0,0

for i in range(predictions.shape[0]):
  if (predictions[i] == answers[i]): correct += 1
  else : incorrect += 1

print(f"Correct : {correct}")
print(f"Incorrect : {incorrect}")
print(f"accuracy : {correct / (correct + incorrect)}")


# %%
predictions = sk_gaussian_kernel.predict(validation_eg)
correct , incorrect = 0,0

for i in range(predictions.shape[0]):
  if (predictions[i] == answers[i]): correct += 1
  else : incorrect += 1

print(f"Correct : {correct}")
print(f"Incorrect : {incorrect}")
print(f"accuracy : {correct / (correct + incorrect)}")

# %% [markdown]
# As can be observed, the accuracy obtained from the scikit learn SVM function is exactly the same as obtained from our implementation. Hence our model has been implemented correctly.

# %%
linear_sv = sk_linear_kernel.support_
gaussian_sv = sk_gaussian_kernel.support_
# print(linear_sv)
sk_set1 = set(linear_sv)
sk_set2 = set(gaussian_sv)
set1 = set( linear_kernel.support_vectors[:, 1] )
set2 = set( gaussian_kernel.support_vectors[:, 1]   )

print(f"no of support vectors for sklearn linear svm is {len(sk_set1)}")
print(f"no of support vectors for sklearn gaussian svm is {len(sk_set2)}")
print(f"no of support vectors for  linear svm is {len(set1)}")
print(f"no of support vectors for  gaussian svm is {len(set2)}")

print(f"no of support vectors common for linear svm are {len(sk_set1 & set1)}")
print(f"no of support vectors common for gaussian svm is {len(sk_set2 & set2)}")


# %% [markdown]
# no of support vectors for sklearn linear svm is 2899
# 
# no of support vectors for sklearn gaussian svm is 3398
# 
# no of support vectors for  linear svm is 2903
# 
# no of support vectors for  gaussian svm is 3453
# 
# no of support vectors common for linear svm are 2899
# 
# no of support vectors common for gaussian svm is 3398
# 
# From the numbers we conclude that the sv used in both of our implmented models are also used by the corresponding sklearn models, along with a few additional sv.

# %%
sk_weight = sk_linear_kernel.coef_
sk_bias = sk_linear_kernel.intercept_

og_weight = linear_kernel.get_weight()
og_bias = linear_kernel.get_bias()

# %%
print(f"bias obtained for sklearn linear svm is {sk_bias}")

print(f"bias obtained for  linear svm is {og_bias}")

# %%
temp = (sk_weight[0] - og_weight )
norm_sk = sk_weight[0].T @ sk_weight[0]
norm = temp.T @ temp
print(math.sqrt(norm/norm_sk) * 100 )

# %% [markdown]
# bias obtained for sklearn linear svm is 0.7251
# bias obtained for  linear svm is 0.7251
# 
# The biases obtained are identical, the weight vectors are also almost identical, the rms error being 0.247 %

# %% [markdown]
# time taken by our linear svm is 80 sec.
# 
# time taken by our gaussian svm is 62 sec.
# 
# time taken by sklearn linear svm is 12 sec
# 
# time taken by sklearn gaussian svm is 7 sec
# 
# Hence there is quite a reduction in the training time when using sklearn !!

# %% [markdown]
# # Multiclass Classification

# %%

class Multi_Class_SVM:
  def __init__(self, X, Y, k, C, gamma):
    self.m = X.shape[0]
    # print(self.m)
    self.X = X
    self.Y = Y
    self.no_of_classes = k
    self.training_datasets = self.split_dataset()
    self.C = C
    self.gamma = gamma
    self.misclassifications = [[[] for i in range(self.no_of_classes)] for j in range(self.no_of_classes)] 


  def split_dataset(self):
    temp = [[] for i in range(self.no_of_classes)]
    for i in range(self.m):
      (temp[self.Y[i]]).append(self.X[i])

    return np.array(temp)


  def train(self):
    self.model = np.full((self.no_of_classes, self.no_of_classes), None, dtype=object)
    for i in range(self.no_of_classes):
      for j in range(self.no_of_classes):
        if (i == j or self.model[i,j] != None): continue
        print(f"training being done for classes {i} and {j}")
        X_ij = np.vstack( ( np.array(self.training_datasets[i]), np.array(self.training_datasets[j]) )  )
        size1 = len(self.training_datasets[i])
        size2 = len(self.training_datasets[j])
        Y_ij = np.zeros( size1 + size2 )
        for k in range(size1 ):
          Y_ij[k] = -1
        for k in range(size1, size1 + size2):
          Y_ij[k] = 1
        # print(X_ij.shape, Y_ij.shape)
        self.model[i,j] = Gaussian_Kernel(X_ij, Y_ij, self.C, self.gamma)
        print("invoking solver now") 
        (self.model[i,j]).solve()
        print("solved")
        (self.model[i,j]).get_support_vectors()
        (self.model[i,j]).get_bias()
        self.model[j, i] = self.model[i,j]
    print("training complete")

  def predict(self, eg):
    scores = np.zeros( (self.no_of_classes, ))
    counts =  np.zeros( (self.no_of_classes, ))
    done = np.zeros((self.no_of_classes, self.no_of_classes))
    for i in range(self.no_of_classes):
      for j in range(self.no_of_classes):
        if (i == j or done[i][j] == 1) : continue
        done[i][j] = 1
        done[j][i] = 1
        # i is -1, j is one
        (prediction, score) = self.model[i,j].predict(eg)
        if (prediction == -1) : prediction = i
        else : prediction = j
        counts[prediction] += 1
        if (score > 0) : scores[prediction] += score
        else: scores[prediction] -= score

    max_count = np.max(counts)
    max_indices = np.where(counts == max_count)[0]
    max_scores = np.array([ (scores[i], i) for i in max_indices])
    sorted_indices = np.argsort(max_scores[:, 1])
    max_scores = max_scores[sorted_indices]
    return (max_scores[-1, 1], max_scores[-1, 0])

  def get_confusion_matrix(self, test_eg, test_ans) :
    size = test_eg.shape[0]
    confusion_matrix = np.zeros( (self.no_of_classes, self.no_of_classes), dtype=int)
    correct , incorrect = 0,0
    for k in range(size):
        (prediction, score) = self.predict(test_eg[k])
        # print(prediction, "\n", test_ans[k]) 
        # print("prediction is ", prediction) 
        self.misclassifications[test_ans[k]][int(prediction)].append(test_eg[k])  
        confusion_matrix[test_ans[k]][int(prediction)] += 1 
        if (test_ans[k] == prediction) : correct += 1
        else : incorrect +=  1

    return (confusion_matrix, (correct / (correct + incorrect)))



# %%
# design matrix
folder_name = "./svm/train/"
no_of_classes = 6
classes = [i for i in range(no_of_classes)]
X_full = tuple()
new_shape = (16, 16)
Y_full = np.array([], dtype = int)
for label in classes:
  for filename in os.listdir(f"{folder_name}{label}"):
        file = os.path.join(f"{folder_name}{label}", filename)
        Z = resize(file, new_shape)
        X_full = X_full + ( Z[None , ...], )
        Y_full = np.append(Y_full, label)

X_full = np.vstack(X_full)

# %%
# design matrix
folder_name = "./svm/val/"
no_of_classes = 6
classes = [i for i in range(no_of_classes)]
X_val_full = tuple()
new_shape = (16, 16)
Y_val_full = np.array([], dtype = int)
for label in classes:
  for filename in os.listdir(f"{folder_name}{label}"):
        file = os.path.join(f"{folder_name}{label}", filename)
        Z = resize(file, new_shape)
        X_val_full = X_val_full + ( Z[None , ...], )
        Y_val_full = np.append(Y_val_full, label)

X_val_full = np.vstack(X_val_full)

# %%
C = 1.0
gamma = 0.001
multiclass_classifier = Multi_Class_SVM(X_full, Y_full, no_of_classes, C, gamma)

# %%
multiclass_classifier.train()

# %%

sk_multiclass_classifier = svm.SVC(C = C, gamma = gamma, kernel = "rbf") 
# , decision_function_shape = "ovr",break_ties = True)
sk_multiclass_classifier.fit(X_full, Y_full)

# %%

def draw_confusion_matrix( confusion_matrix, name):
    
        correct = 0 
        total = 0 
        fig, ax = plt.subplots(figsize=(10,10))
        ax.matshow(confusion_matrix, cmap=plt.cm.Blues, alpha=0.3) 
        max_diag = 0 
        max_diag_label = 0
        for i in range(confusion_matrix.shape[0]):
            for j in range(confusion_matrix.shape[1]):
                total += confusion_matrix[i,j]
                if (i == j) : 
                    correct += confusion_matrix[i,j]
                    if (max_diag < confusion_matrix[i,j]):
                        max_diag = confusion_matrix[i,j]
                        max_diag_label = i
    
                ax.text(x=j, y=i,s= confusion_matrix[i, j], va='center', ha='center', size='xx-large')
        
        plt.xlabel('Predictions', fontsize=18)
        plt.ylabel('Actuals', fontsize=18)
        plt.title('Confusion Matrix', fontsize=18)
        plt.savefig(f"confusion_matrix_{name}.png")
        # plt.show()
        print(f"accuracy is {correct/total}")
        print(f"label with max diagonal is {max_diag_label}")




# %%

(confusion_matrix, accuracy) = multiclass_classifier.get_confusion_matrix(X_val_full, Y_val_full)
print(f"accuracy is {accuracy}")
draw_confusion_matrix(confusion_matrix, "our_multiclass")

# %%
# visualising error images 
cnt = 0 
for img in multiclass_classifier.misclassifications[4][2]:
    if (cnt > 2): break
    cnt += 1 
    img = img * 255.0
    img = img.reshape((16,16,3))
    cv2.imwrite(f"img_4_2_{cnt}.png", img)
cnt = 0 
for img in multiclass_classifier.misclassifications[4][3]:
    if (cnt > 2): break
    cnt += 1 
    img = img * 255.0
    img = img.reshape((16,16,3))
    cv2.imwrite(f"img_4_3_{cnt}.png", img)
cnt = 0 
for img in multiclass_classifier.misclassifications[0][3]:
    if (cnt > 2): break
    cnt += 1 
    img = img * 255.0
    img = img.reshape((16,16,3))
    cv2.imwrite(f"img_0_3_{cnt}.png", img)
cnt = 0 
for img in multiclass_classifier.misclassifications[1][5]:
    if (cnt > 2): break
    cnt += 1 
    img = img * 255.0
    img = img.reshape((16,16,3))
    cv2.imwrite(f"img_1_5_{cnt}.png", img)


# resized_weight = weight * 255.0
# resized_weight = resized_weight.reshape( (16, 16, 3))
# cv2.imwrite(f"weight_{svm_name}.png", resized_weight)

# %% [markdown]
# set 0 has images of houses and buildings, 1 has of forests and greenery, 2 has of ice and glaciers, 3 has of cloudy sky and mountains, 4 has of coasts and oceans,  5 of cities. 
# 
# Of these the most frequent pair of misclassified classes are (first is actual, second is prediction) : 
# (4,2), (4,3), (0,3), (1,5). 
# Based on the above observations, it is easy to make sense of the visualisations of the misclassified examples as showm. For eg - images of 4 and 2 both are predominantly white and blue, thus they are misclassified as each other the most. 
# Similiar logic holds for 4 and 3, and so on. 

# %%
predictions = sk_multiclass_classifier.predict(X_val_full)
correct , incorrect = 0,0

# print(predictions.shape[0])
sk_confusion_matrix = np.array( [[0 for i in range(no_of_classes)] for j in range(no_of_classes)] ) 

for i in range(predictions.shape[0]):
  # print(predictions[i], Y_val_full[i])
  sk_confusion_matrix[Y_val_full[i]][predictions[i]] += 1
  if (predictions[i] == Y_val_full[i]): correct += 1
  else : incorrect += 1

print(f"Correct : {correct}")
print(f"Incorrect : {incorrect}")
print(f"accuracy : {correct / (correct + incorrect)}")
draw_confusion_matrix(sk_confusion_matrix, "sklearn_multiclass")


# %% [markdown]
# Each class has 200 samples in the validation set. The confusion matrices obtained from our implementation and from sklearn are largely similiar, differing by only a small amt. 
# We also observe that class 1 is classified most correctly, whereas class 4 is the least correctly classifed class. 
# The largest off diagonal entry is (4,2), which means that most misclassifications have been done for class 4 where it was confused for class 2 (58 examples). Similarly we observe that class 0 is misclassified most frequently as class 5, 1 as 5, 3 as 2, 4 as 2, 5 as 0. 
# 
# 

# %% [markdown]
# # Cross Validation

# %%
k = 5 
def get_splits(x,y,k):
    m = x.shape[0]
    splits = []
    for i in range(k):
        splits.append( (x[int(i*m/k) : int((i+1)*m/k)], y[int(i*m/k) : int((i+1)*m/k)]) )
    
    val_splits = [] 
    for i in range(k):
        val_split = [] 
        for j in range(k):
            if (i == j): continue
            val_split.append(splits[j]) 
        x_tup = [] 
        y_tup = []
        for s in val_split:
            x_tup.append(s[0])
            y_tup.append(s[1]) 
        x_tup = np.vstack(tuple(x_tup)) 
        y_tup = np.hstack(tuple(y_tup))
        val_splits.append((x_tup, y_tup)) 
    return (val_splits, (splits)) 

# %%
perm = np.random.permutation([i for i in range(X_full.shape[0])]) 
combined_x = np.zeros(X_full.shape)
combined_y = np.zeros(Y_full.shape)
for i in range(X_full.shape[0]):
    combined_x[i] = X_full[perm[i]]
    combined_y[i] = Y_full[perm[i]] 

(splits, og_splits) = get_splits(combined_x, combined_y, k)
for split in splits:
    print(split[0].shape, split[1].shape)


# %%
C_arr = np.array([1e-05, 1e-03, 1, 5, 10]) 
cross_validation_ac = np.zeros(C_arr.shape)
validation_ac = np.zeros(C_arr.shape)
k = 5 
for j in range( C_arr.shape[0]):
    avg = 0 
    for i in range(k):
        correct = 0
        incorrect = 0 
        sk_model = svm.SVC(C = C_arr[j] , kernel = "rbf", gamma = 0.001) 
        sk_model.fit(splits[i][0], splits[i][1]) 
        predictions = sk_model.predict(og_splits[i][0]) 

        for w in range(og_splits[i][1].shape[0]):
            if (predictions[w] == og_splits[i][1][w]): correct += 1
            else : incorrect += 1
        print(f"for model {j} and dataset {i}, correct : {correct} and incorrect : {incorrect}") 
        accuracy = correct / (correct + incorrect) 
        avg += accuracy
    avg /= k 
    cross_validation_ac[j] = avg 
    sk_full_model = svm.SVC(C = C_arr[j] , kernel = "rbf", gamma = 0.001) 
    sk_full_model.fit(combined_x,  combined_y) 
    predictions = sk_full_model.predict(X_val_full) 
    correct = 0
    incorrect = 0
    for w in range(Y_val_full.shape[0]):
        if (predictions[w] == Y_val_full[w]): correct += 1
        else : incorrect += 1
    print(f"for model {j} validation, correct : {correct} and incorrect : {incorrect}") 
    accuracy = correct / (correct + incorrect)
    validation_ac[j] = accuracy
print(cross_validation_ac)
print(validation_ac)

# %%
for j in range( C_arr.shape[0]):
    correct = 0
    incorrect = 0 
    sk_model = svm.SVC(C = C_arr[j] , kernel = "rbf", gamma = 0.001) 
    sk_model.fit(X_full, Y_full) 
    predictions = sk_model.predict(X_val_full)
    for w in range(Y_val_full.shape[0]):
        if (predictions[w] == Y_val_full[w]): correct += 1
        else : incorrect += 1
    print(f"for model {j} on full dataset, correct : {correct} and incorrect : {incorrect}") 
    accuracy = correct / (correct + incorrect) 
    

# %%

def plot_line(x, y, xlabel, ylabel, title):
    plt.plot(x, y, marker='o', label = title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    # plt.title(title)
    # plt.savefig(f"{name}.png")
    # plt.show()

plot_line( np.log(C_arr), cross_validation_ac, "log C", "accuracy", "cross_validation_accuracy")
plot_line( np.log(C_arr), validation_ac, "log C", "accuracy", "validation_accuracy") 
plt.title("cross_validation_accuracy vs validation_accuracy")
plt.legend()
plt.savefig("cross_validation_accuracy vs validation_accuracy.png")
plt.show()

# %% [markdown]
# We observe that the value of C which gives the best cross validation accuracy is also the same which gives the
# best validation accuracy. 
# In fact as the cross validation accuracy (CVA) increases, the validation accuracy (VA) also increases with C. 
# 
# Thus cross validation accuracy serves as a good measure of the generalized accuracy, i.e we can say for our case that if a model with C = C1 has better cross validation accuracy than model with C = C2, then its generalized accuracy is also expected to be better.


