import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys

class GPA:
    def __init__(self,scale_factors=[],stab_energies=[],avoided_crossings=[],av_roots=[]):
        self.scale_factors=np.array(scale_factors)
        self.stab_energies=np.array(stab_energies)
        self.avoided_crossings=np.array(avoided_crossings)
        self.av_roots=np.array(av_roots)

    def get_data_from_file(self,filename=''):
        file1=open(filename,'r')
        lines=file1.readlines()
        file1.close()
        sfs=[]
        es=[]
        for i in range(len(lines[0].strip().split(','))-1):
            es.append([])
        for line in lines:
            line_array=line.strip().split(',')
            sfs.append(float(line_array[0]))
            for i in range(len(line_array)-1):
                es[i].append(float(line_array[i+1]))
        self.scale_factors=np.array(sfs)
        self.stab_energies=np.array(es)
        return np.array(sfs),np.array(es)

    def make_stab_plot(self):
        sfs=self.scale_factors
        for i in self.stab_energies:
            plt.plot(sfs,i)
        plt.show()

    def detect_avoided_crossings(self,each_side=10,plot=False):
        sfs=self.scale_factors
        acs=[]
        for i in range(len(self.stab_energies)-1):
            all_diffs=[]
            k=0
            for j in range(len(self.stab_energies[i])):
                diff=self.stab_energies[i+1,j]-self.stab_energies[i,j]
                all_diffs.append(diff)
                if j==0:
                    continue
                if j==1 and all_diffs[-1]>all_diffs[-2]:
                    k=1
                    continue
                if k==1 and all_diffs[-2]>all_diffs[-1]:
                    k=0
                    continue
                if k==0 and all_diffs[-1]>all_diffs[-2]:
                    k=1
                    acs.append(np.array([sfs[max(0,(j-1)-each_side):min(len(sfs),j+each_side)],self.stab_energies[i,max(0,(j-1)-each_side):min(len(sfs),j+each_side)],self.stab_energies[i+1,max(0,(j-1)-each_side):min(len(sfs),j+each_side)]]))
                    continue
        if plot:
            for i in acs:
                plt.scatter(i[0],i[1])
                plt.scatter(i[0],i[2])
            plt.show()
        self.avoided_crossings=acs
        return acs

    def gpa(self,porder=0,qorder=1,rorder=2,nr_tol=1.0e-5,sqrt_num_in_points=10,max_nr_iter=100,verbose=0.):
        def func(X,porder0,qorder0,rorder0,params0):
            x,e=X
            ppoly=1.
            for j in range(porder0):
                ppoly+=(params0[j]*(x**(j+1)))
            qpoly=0.
            for j in range(qorder0+1):
                qpoly+=(params0[j+porder]*(x**j))
            rpoly=0.
            for j in range(rorder+1):
                rpoly+=(params0[j+porder+qorder+1]*(x**j))
            return ((e**2)*ppoly)+(e*qpoly)+rpoly
        def dfuncdx(X,porder0,qorder0,rorder0,params0):
            x,e=X
            ppoly=0.
            for j in range(porder0):
                ppoly+=((j+1)*params0[j]*(x**j))
            qpoly=0.
            for j in range(qorder0+1):
                qpoly+=(j*params0[j+porder]*(x**(j-1)))
            rpoly=0.
            for j in range(rorder+1):
                rpoly+=(j*params0[j+porder+qorder+1]*(x**(j-1)))
            return ((e**2)*ppoly)+(e*qpoly)+rpoly
        def dfuncde(X,porder0,qorder0,params0):
            x,e=X
            ppoly=1.
            for j in range(porder0):
                ppoly+=(params0[j]*(x**(j+1)))
            qpoly=0.
            for j in range(qorder0+1):
                qpoly+=(params0[j+porder]*(x**j))
            return (2*e*ppoly)+qpoly
        def ddfuncddx(X,porder0,qorder0,rorder0,params0):
            x,e=X
            ppoly=0.
            for j in range(porder0):
                ppoly+=((j+1)*j*params0[j]*(x**(j-1)))
            qpoly=0.
            for j in range(qorder0+1):
                qpoly+=(j*(j-1)*params0[j+porder]*(x**(j-2)))
            rpoly=0.
            for j in range(rorder+1):
                rpoly+=(j*(j-1)*params0[j+porder+qorder+1]*(x**(j-2)))
            return ((e**2)*ppoly)+(e*qpoly)+rpoly
        def ddfuncdxde(X,porder0,qorder0,params0):
            x,e=X
            ppoly=0.
            for j in range(porder0):
                ppoly+=((j+1)*params0[j]*(x**j))
            qpoly=0.
            for j in range(qorder0+1):
                qpoly+=(j*params0[j+porder]*(x**(j-1)))
            return (2*e*ppoly)+qpoly
        def e_plus(x,porder0,qorder0,rorder0,params0):
            ppoly=1.
            for j in range(porder0):
                ppoly+=(params0[j]*(x**(j+1)))
            qpoly=0.
            for j in range(qorder0+1):
                qpoly+=(params0[j+porder]*(x**j))
            rpoly=0.
            for j in range(rorder+1):
                rpoly+=(params0[j+porder+qorder+1]*(x**j))
            return (-qpoly+np.sqrt((qpoly**2)-(4*ppoly*rpoly),dtype='complex128'))/(2*ppoly)
        def e_minus(x,porder0,qorder0,rorder0,params0):
            ppoly=1.
            for j in range(porder0):
                ppoly+=(params0[j]*(x**(j+1)))
            qpoly=0.
            for j in range(qorder0+1):
                qpoly+=(params0[j+porder]*(x**j))
            rpoly=0.
            for j in range(rorder+1):
                rpoly+=(params0[j+porder+qorder+1]*(x**j))
            return (-qpoly-np.sqrt((qpoly**2)-(4*ppoly*rpoly),dtype='complex128'))/(2*ppoly)
        av_roots=[]
        for i in self.avoided_crossings:
            sfs=np.array(list(i[0])+list(i[0]))
            es=np.array(list(i[1])+list(i[2]))
            ydata=np.zeros(2*len(i[0]))
            params=np.array([1. for _ in range(porder+qorder+rorder+2)])
            popt,pcov=curve_fit(lambda X,*params:func(X,porder,qorder,rorder,params),(sfs,es),ydata,p0=params)
            step=(max(i[0][0],i[0][-1])-min(i[0][0],i[0][-1]))/(sqrt_num_in_points-1.)
            test_grid=np.arange(min(i[0][0],i[0][-1]),max(i[0][0],i[0][-1])+(step/2.),step)
            parity=[-1,1]
            roots=[]
            for j in test_grid:
                for k in test_grid:
                    for par in parity:
                        xguess=complex(j,par*k)
                        pguess=np.array([xguess,e_plus(xguess,porder,qorder,rorder,popt)])
                        mguess=np.array([xguess,e_minus(xguess,porder,qorder,rorder,popt)])
                        pconverged=False
                        mconverged=False
                        for l in range(max_nr_iter):
                            pfvec=np.array([func((pguess[0],pguess[1]),porder,qorder,rorder,popt),dfuncdx((pguess[0],pguess[1]),porder,qorder,rorder,popt)])
                            mfvec=np.array([func((mguess[0],mguess[1]),porder,qorder,rorder,popt),dfuncdx((mguess[0],mguess[1]),porder,qorder,rorder,popt)])
                            pkmat=np.array([[dfuncdx((pguess[0],pguess[1]),porder,qorder,rorder,popt),dfuncde((pguess[0],pguess[1]),porder,qorder,popt)],[ddfuncddx((pguess[0],pguess[1]),porder,qorder,rorder,popt),ddfuncdxde((pguess[0],pguess[1]),porder,qorder,popt)]])
                            mkmat=np.array([[dfuncdx((mguess[0],mguess[1]),porder,qorder,rorder,popt),dfuncde((mguess[0],mguess[1]),porder,qorder,popt)],[ddfuncddx((mguess[0],mguess[1]),porder,qorder,rorder,popt),ddfuncdxde((mguess[0],mguess[1]),porder,qorder,popt)]])
                            invpkmat=np.linalg.inv(pkmat)
                            invmkmat=np.linalg.inv(mkmat)
                            pdelta=-np.dot(invpkmat,pfvec)
                            mdelta=-np.dot(invmkmat,mfvec)
                            pguess+=pdelta
                            mguess+=mdelta
                            if np.sum(np.abs(pdelta))<nr_tol:
                                pconverged=True
                            if np.sum(np.abs(mdelta))<nr_tol:
                                mconverged=True
                            if pconverged and mconverged:
                                break
                        if pconverged:
                            not_same=True
                            for m in roots:
                                same_test=np.sum(np.abs(m-pguess))
                                if same_test<nr_tol:
                                    not_same=False
                                    break
                            if not_same:
                                roots.append(pguess)
                        if mconverged:
                            not_same=True
                            for m in roots:
                                same_test=np.sum(np.abs(m-mguess))
                                if same_test<nr_tol:
                                    not_same=False
                                    break
                            if not_same:
                                roots.append(mguess)
            av_roots.append(np.array(roots))
            if verbose>=2.:
                print('Avoided Crossing:')
                print(i)
                print('Roots:')
                print(av_roots[-1])
                print('')
        self.av_roots=av_roots
        return av_roots
 
if __name__ == "__main__":
    test=GPA()
    sfs,es=test.get_data_from_file(filename='test.txt')
    print(sfs)
    print(es)
    test.make_stab_plot()
    test.detect_avoided_crossings(each_side=20,plot=False)
    test.gpa(porder=3,qorder=4,rorder=5,nr_tol=1.0e-5,sqrt_num_in_points=10,max_nr_iter=100,verbose=4.)
    test.detect_avoided_crossings(each_side=20,plot=True)


