import numpy as np
from typing import callable

class svm:
    def __init__(self):
        pass
    
    def fit(self, y, x, alphas, kernel = svm.linear_kernel,c: float = 1, tol: float = 1e-3, eps: float = 1e-3):

        # instanciate the parameters
        self.y = y
        self.x = x
        self.alphas = alphas
        self.c = c
        self.tol = tol
        self.eps = eps
        self.ker_matrix = kernel(self.x, self.x)
        # I still need to initialize the self.alphas and the threshold b
        self.err_cache = ( self.y * self.alphas ) @ self.ker_matrix - b

        num_changed = 0
        examin_all = 1

        while num_changed > 0 or examin_all:
            num_changed = 0

            if examin_all:
                for i2 in range(len(targets)):
                    num_changed += examin_example(i2)

            else:
                for i2 in non_bd:
                    num_changed += examin_example(i2)

            if examin_all == 1:
                examin_all = 0
            elif num_changed == 0:
                examin_all = 1


    def examin_example(self, i2):
        y2 = self.y[i2]
        # print("y2", y2)
        a2 = self.alphas[i2]
        e2 = self.err_cache[0][i2] 
        # print("e2: ", e2)
        r2 = e2 * y2
        # print(r2)
        # print(type(r2))

        nb_idx = find_nb(targets, alphas, kernel, c, tol)
        # e_nb = predict(training[non_bounds,:]) - targets[non_bounds]

        if (r2 < -tol and a2 < c) or (r2 > tol and a2 > 0):
            # choose the lagrange multipliers that do not belongs to the boundry of [0, c]
            if len(nb_idx) > 1:
                i1 = second_choice(err_cache, e2)
                i1 = training[nb_idx[i1]]
                if take_step(i1, i2):
                    return 1

            # if no ggood candidate for i1 is found in the previous loop, it loops over le non  boud lagrange multipliers
            for i1 in np.random.permutation(nb_idx):
                if take_step(i1, i2):
                    return 1
            # if there aren't non bound lagrange multipliers it loops over all the multipliers
            for i1 in np.random.permutation(len(targets)):
                if take_step(i1, i2):
                    return 1

        return 0

    def take_step(i1, i2):

        if i1 == i2:
            return 0
        
        x1 = self.x[i1]
        x2 = self.x[i2]
        y1 = self.y[i1]
        y2 = self.y[i2]
        a1 = self.alphas[i1]
        a2 = self.alphas[i2]
        e1 = self.err_cache[i1]
        e2 = self.err_cache[i2]
        
        s = y1 * y2

        # Compute the boundaries
        l, h = self.compute_boundaries(a1, a2, y1, y2)    
        
        # If the boundaries are equal I stop the step
        if np.abs(l - h) < self.tol:
            return 0 

        # Compute the new Lagrange multipliers
        a1_new, a2_new = self.compute_new_a(i1, i2, a1, a2, s, l, h, e1, e2)

        # Stop the step is the new a2_new is not significantly different from the old one
        if np.abs(a2_new - a2) < self.eps * (a2_new + a2 + self.eps):
            return 0

        # update threshold to reflect change in lagrange multipliers
        b_new = self.compute_threshold(i1, i2, y1, y2, e1, e2, a1, a1_new, a2, a2_new, b)
        
        # update error cache using new lagrange multipliers
        self.err_cache = self.update_err_cache(i1, i2, y1, y2, a1, a2, a1_new, a2_new, b, b_new)

        # Update the threshold value
        b = b_new

        # update the a1 and a2
        self.alphas[i1] = a1_new
        self.alphas[i2] = a2_new

        # update weight vector to reflect change in a1, a2 (no weight vector for the moment)

        return 1

    def compute_err_cache(self):
        return 
    def update_err_cache(self, i1, i2, y1, y2, a1, a2, a1_new, a2_new, b, b_new):
        diff = (a1_new - a1) * y1 * self.ker_matrix[i1, :] + (a2_new - a_2) * y2 * self.ker_matrix[i2, :] - (b_new - b)
        return self.err_cache + diff



    def compute_threshold(self,i1, i2, y1, y2, e1, e2, a1, a1_new, a2, a2_new, b):

        if self.tol < a1 < c - self.tol:
            b1 = e1 + y1 * (a1_new - a1) * self.ker_matrix(i1, i2) + y2 * (a2_new - a2) * self.ker_matrix(i1, i2) + b
            return b1

        if self.tol < a2 < c - self.tol:
            b2 = e2 + y1 * (a1_new - a1) * self.ker_matrix(i1, i2) + y2 * (a2_new - a2) * self.ker_matrix(i1, i2) + b
            return b2

        b1 = e1 + y1 * (a1_new - a1) * self.ker_matrix(i1, i2) + y2 * (a2_new - a2) * self.ker_matrix(i1, i2) + b
        b2 = e2 + y1 * (a1_new - a1) * self.ker_matrix(i1, i2) + y2 * (a2_new - a2) * self.ker_matrix(i1, i2) + b
        return (b1 + b2) / 2

    def compute_new_a(self,i1, i2, a1: float, a2: float,s: int, L: float, H: float, eta: float, e1, e2):
        """
        Compute the Lagrange multipliers a1_cl and a2_cl

        Arguments:
            a1: float, the current Lagrange multiplier
            a2: float, the current Lagrange multiplier
            s: int, the product of the targets associated to the Lagrange multipliers
            L: float, the lower boundary of a2
            H: float, the higher boundary of a2
            eta: float, the second derivative of the objective function
            err_1: float, the error of the training example associated to y_1
            err_2: float, the error of the training example associated to y_2

        Returns:
            The new optimized Lagrange multipliers a1_new, a2_new.
        """
        
        # Compute eta, the second derivative of the objective function along the line defined by a1 and a2
        eta = self.compute_eta(i1, i2)

        if eta > 0:
            a2_new = a2 + (y_2 * (err_1 - err_2)) / eta
            a2_new = H if a2_new >= H else a2_new if L < a2_new < H else L
            a1_new = a1 + s * (a2 - a2_new)
            return a1_new, a2_new
        
        
        f1 = y1 * (err_1 + b) - a1 * self.ker_matrix[i1, i1] - s * a2 * self.ker_matrix[i1, i2]
        f2 = y2 * (err_2 + b) - s * a1 * self.ker_matrix[i1, i2] - a2 * self.ker_matrix[i2, i2]
        L1 = a1 + s * (a2 - L)
        H1 = a1 + s * (a2 - H)
        ob_func_L = L1 * f1 + L * f2 + 0.5 * L1**2 * self.ker_matrix[i1, i1] + 0.5 * L**2 * self.ker_matrix[i2, i2] + s * L * L1 * self.ker_matrix[i1, i2]
        ob_func_H = H1 * f1 + H * f2 + 0.5 * H1**2 * self.ker_matrix[i1, i1] + 0.5 * H**2 * self.ker_matrix[i2, i2] + s * H * H1 * self.ker_matrix[i1, i2]

        if ob_func_L > ob_func_H - self.eps:
            a2_new = L
        elif ob_func_L > ob_func_H + self.eps:
            a2_new = H
        else:
            a2_new = a2

        a1_new = a1 + s * (a2 - a2_new)
        return a1_new, a2_new


    def compute_boundaries(self, a1: float, a2: float, y1: int, y2: int):
        """
        compute the boundaries of the lagrangian multiplier a2.

        arguments:
            a1: float, a lagrangian multiplier
            a2: float, a lagrangian multiplier
            y1: int, the target associated to a1
            y2: int, the target associated to a2
            c: float, the control constant of the objective

        returns:
            l: float, the lower boundary for a2
            h: float, the highest boundary for a2
            
        """
        if y1 != y2:
            l = np.max([0, a2 - a1])
            h = np.min([c, c + a2 - a1])

        else:
            l = np.max([0, a2 + a1 - c])
            h = np.min([c,a2 + a1])

        return l, h

    def compute_eta(self, i1, i2):
        return self.ker_matrix[i1,i1] + self.ker_matrix[i2,i2] - 2 * self.ker_matrix[i1,i2]

    def compute_scores(self, x, trainings, targets, alphas, kernel, b):
        yalphas = targets * alphas
        # i should define the kernel matrix of the training set once, because to fit the model i always use the same. 
        return yalphas.reshape(1, -1) @ kernel(x, x) + b

    @staticmethod
    def linear_kernel(v: np.ndarray, w: np.ndarray) -> float:
        """
        Linear kernel implementation.

        Arguments:
            v: a (m,) vector or a family of n vector of dimension m as a (n,m) matrix
            w: a (m,) vector or a family of n vector of dimension m as a (n,m) matrix
        
        Returns:
            the linear kernel:
                - It computes the dot product of v and w.T.

        """
        return v @ w.T

