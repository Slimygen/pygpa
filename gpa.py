import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import sys

class GPA:
    def __init__(self,scale_factors=[],stab_energies=[],avoided_crossings=[],av_roots=[]):
        self.scale_factors=np.array(scale_factors)
        self.stab_energies=np.array(stab_energies)
        self.avoided_crossings=list(avoided_crossings)
        self.av_roots=list(av_roots)

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
        full_len=len(es[0])
        to_remove=[]
        for i in range(len(es)):
            if len(es[i])<full_len:
                to_remove.append(i)
        for i in range(len(to_remove)):
            es.remove(es[to_remove[len(to_remove)-(i+1)]])
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
        avs=[]
        acs=self.avoided_crossings
        avrs=self.av_roots
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
                    avs.append(acs[-1])
                    avrs.append(np.array([]))
                    continue
        if plot:
            for i in avs:
                plt.scatter(i[0],i[1])
                plt.scatter(i[0],i[2])
            plt.show()
        self.avoided_crossings=acs
        self.av_roots=avrs
        return avs

    def gpa(self,wac=1,orders=[2,1,0],nr_tol=1.0e-5,diff_sol_tol=1.0e-2,sqrt_num_in_points=10,max_nr_iter=100,gd_step=0.01,gd_num_its=100,gd_tol=1.e-5,verbose=0.):
        def func(X,orders0,params0):
            x,e=X[0],X[1]
            len_orders0=len(orders0)
            if len_orders0==0:
                return 0.
            order0=orders0[-1]
            rest_orders0=[]
            if len_orders0>1:
                rest_orders0=orders[:-1]
            tot=1.
            for j in range(order0):
                tot+=(params0[j]*(x**(j+1)))
            tot*=(e**(len_orders0-1))
            temptot=0.
            prev_order=order0
            for j in range(len(rest_orders0)):
                temptot=0.
                for i in range(rest_orders0[len(rest_orders0)-(j+1)]+1):
                    temptot+=(params0[i+prev_order]*(x**i))
                prev_order+=(rest_orders0[len(rest_orders0)-(j+1)]+1)
                temptot*=(e**(len_orders0-(2+j)))
                tot+=temptot
            return tot
        def dfuncdx(X,orders0,params0):
            x,e=X[0],X[1]
            len_orders0=len(orders0)
            if len_orders0==0:
                return 0.
            order0=orders0[-1]
            rest_orders0=[]
            if len_orders0>1:
                rest_orders0=orders[:-1]
            tot=0.
            for j in range(order0):
                tot+=((j+1)*params0[j]*(x**(j)))
            tot*=(e**(len_orders0-1))
            temptot=0.
            prev_order=order0
            for j in range(len(rest_orders0)):
                temptot=0.
                for i in range(rest_orders0[len(rest_orders0)-(j+1)]+1):
                    temptot+=(i*params0[i+prev_order]*(x**(i-1)))
                prev_order+=(rest_orders0[len(rest_orders0)-(j+1)]+1)
                temptot*=(e**(len_orders0-(2+j)))
                tot+=temptot
            return tot
        def dfuncde(X,orders0,params0):
            x,e=X[0],X[1]
            len_orders0=len(orders0)
            if len_orders0==0:
                return 0.
            order0=orders0[-1]
            rest_orders0=[]
            if len_orders0>1:
                rest_orders0=orders[:-1]
            tot=1.
            for j in range(order0):
                tot+=(params0[j]*(x**(j+1)))
            tot*=((len_orders0-1)*(e**(len_orders0-2)))
            temptot=0.
            prev_order=order0
            for j in range(len(rest_orders0)):
                temptot=0.
                for i in range(rest_orders0[len(rest_orders0)-(j+1)]+1):
                    temptot+=(params0[i+prev_order]*(x**i))
                prev_order+=(rest_orders0[len(rest_orders0)-(j+1)]+1)
                temptot*=((len_orders0-(2+j))*(e**(len_orders0-(3+j))))
                tot+=temptot
            return tot
        def ddfuncddx(X,orders0,params0):
            x,e=X[0],X[1]
            len_orders0=len(orders0)
            if len_orders0==0:
                return 0.
            order0=orders0[-1]
            rest_orders0=[]
            if len_orders0>1:
                rest_orders0=orders[:-1]
            tot=1.
            for j in range(order0):
                tot+=((j+1)*j*params0[j]*(x**(j-1)))
            tot*=(e**(len_orders0-1))
            temptot=0.
            prev_order=order0
            for j in range(len(rest_orders0)):
                temptot=0.
                for i in range(rest_orders0[len(rest_orders0)-(j+1)]+1):
                    temptot+=(i*(i-1)*params0[i+prev_order]*(x**(i-2)))
                prev_order+=(rest_orders0[len(rest_orders0)-(j+1)]+1)
                temptot*=(e**(len_orders0-(2+j)))
                tot+=temptot
            return tot
        def ddfuncdxde(X,orders0,params0):
            x,e=X[0],X[1]
            len_orders0=len(orders0)
            if len_orders0==0:
                return 0.
            order0=orders0[-1]
            rest_orders0=[]
            if len_orders0>1:
                rest_orders0=orders[:-1]
            tot=1.
            for j in range(order0):
                tot+=((j+1)*params0[j]*(x**(j)))
            tot*=((len_orders0-1)*(e**(len_orders0-2)))
            temptot=0.
            prev_order=order0
            for j in range(len(rest_orders0)):
                temptot=0.
                for i in range(rest_orders0[len(rest_orders0)-(j+1)]+1):
                    temptot+=(i*params0[i+prev_order]*(x**(i-1)))
                prev_order+=(rest_orders0[len(rest_orders0)-(j+1)]+1)
                temptot*=((len_orders0-(2+j))*(e**(len_orders0-(3+j))))
                tot+=temptot
            return tot
        def second_order_e_1(x,porder0,qorder0,rorder0,params0):
            ppoly=1.
            for j in range(porder0):
                ppoly+=(params0[j]*(x**(j+1)))
            qpoly=0.
            for j in range(qorder0+1):
                qpoly+=(params0[j+porder0]*(x**j))
            rpoly=0.
            for j in range(rorder0+1):
                rpoly+=(params0[j+porder0+qorder0+1]*(x**j))
            return (-qpoly+np.sqrt((qpoly**2)-(4*ppoly*rpoly),dtype='complex128'))/(2*ppoly)
        def second_order_e_2(x,porder0,qorder0,rorder0,params0):
            ppoly=1.
            for j in range(porder0):
                ppoly+=(params0[j]*(x**(j+1)))
            qpoly=0.
            for j in range(qorder0+1):
                qpoly+=(params0[j+porder0]*(x**j))
            rpoly=0.
            for j in range(rorder0+1):
                rpoly+=(params0[j+porder0+qorder0+1]*(x**j))
            return (-qpoly-np.sqrt((qpoly**2)-(4*ppoly*rpoly),dtype='complex128'))/(2*ppoly)
        def third_order_e_1(x,porder0,qorder0,rorder0,sorder0,params0):
            ppoly=1.
            for j in range(porder0):
                ppoly+=(params0[j]*(x**(j+1)))
            qpoly=0.
            for j in range(qorder0+1):
                qpoly+=(params0[j+porder0]*(x**j))
            rpoly=0.
            for j in range(rorder0+1):
                rpoly+=(params0[j+porder0+qorder0+1]*(x**j))
            spoly=0.
            for j in range(sorder0+1):
                spoly+=(params0[j+porder0+qorder0+rorder0+1]*(x**j))
            del0=(qpoly**2)-(3.*ppoly*rpoly)
            del1=(2.*(qpoly**3))+(-9.*ppoly*qpoly*rpoly)+(27.*(ppoly**2)*spoly)
            C=np.power((del1+np.sqrt((del1**2)-(4*(del0**3)),dtype='complex128'))/2.,1./3.,dtype='complex128')
            if C==0:
                C=np.power((del1-np.sqrt((del1**2)-(4*(del0**3)),dtype='complex128'))/2.,1./3.,dtype='complex128')
            if C==0:
                dzoc=0
            else:
                dzoc=del0/C
            return (qpoly+C+dzoc)/(-3.*ppoly)
        def third_order_e_2(x,porder0,qorder0,rorder0,sorder0,params0):
            ppoly=1.
            for j in range(porder0):
                ppoly+=(params0[j]*(x**(j+1)))
            qpoly=0.
            for j in range(qorder0+1):
                qpoly+=(params0[j+porder0]*(x**j))
            rpoly=0.
            for j in range(rorder0+1):
                rpoly+=(params0[j+porder0+qorder0+1]*(x**j))
            spoly=0.
            for j in range(sorder0+1):
                spoly+=(params0[j+porder0+qorder0+rorder0+1]*(x**j))
            del0=(qpoly**2)-(3.*ppoly*rpoly)
            del1=(2.*(qpoly**3))+(-9.*ppoly*qpoly*rpoly)+(27.*(ppoly**2)*spoly)
            C=np.power((del1+np.sqrt((del1**2)-(4*(del0**3)),dtype='complex128'))/2.,1./3.,dtype='complex128')
            if C==0:
                C=np.power((del1-np.sqrt((del1**2)-(4*(del0**3)),dtype='complex128'))/2.,1./3.,dtype='complex128')
            if C==0:
                dzoc=0
            else:
                dzoc=del0/C
            epsilon=(-1.+np.sqrt(-3.,dtype='complex128'))/2.
            return (qpoly+(epsilon*C)+(dzoc/epsilon))/(-3.*ppoly)
        def third_order_e_3(x,porder0,qorder0,rorder0,sorder0,params0):
            ppoly=1.
            for j in range(porder0):
                ppoly+=(params0[j]*(x**(j+1)))
            qpoly=0.
            for j in range(qorder0+1):
                qpoly+=(params0[j+porder0]*(x**j))
            rpoly=0.
            for j in range(rorder0+1):
                rpoly+=(params0[j+porder0+qorder0+1]*(x**j))
            spoly=0.
            for j in range(sorder0+1):
                spoly+=(params0[j+porder0+qorder0+rorder0+1]*(x**j))
            del0=(qpoly**2)-(3.*ppoly*rpoly)
            del1=(2.*(qpoly**3))+(-9.*ppoly*qpoly*rpoly)+(27.*(ppoly**2)*spoly)
            C=np.power((del1+np.sqrt((del1**2)-(4*(del0**3)),dtype='complex128'))/2.,1./3.,dtype='complex128')
            if C==0:
                C=np.power((del1-np.sqrt((del1**2)-(4*(del0**3)),dtype='complex128'))/2.,1./3.,dtype='complex128')
            if C==0:
                dzoc=0
            else:
                dzoc=del0/C
            epsilon=(-1.+np.sqrt(-3.,dtype='complex128'))/2.
            return (qpoly+((epsilon**2)*C)+(dzoc/(epsilon**2)))/(-3.*ppoly)
        i=self.avoided_crossings[wac-1]
        sfs=[]
        es=[]
        for j in range(len(i)-1):
            sfs+=list(i[0])
            es+=list(i[j+1])
        sfs=np.array(sfs)
        es=np.array(es)
        ydata=np.zeros((len(i)-1)*len(i[0]))
        params=np.array([1. for _ in range(np.sum(orders)+len(orders)-1)])
        popt,pcov=curve_fit(lambda X,*params:func(X,orders,params),(sfs,es),ydata,p0=params)
        step=(np.amax(sfs)-np.amin(sfs))/(sqrt_num_in_points-1.)
        test_grid=np.arange(min(i[0][0],i[0][-1]),max(i[0][0],i[0][-1])+(step/2.),step)
        if len(orders)>4:
            e_step=(np.amax(es)-np.amin(es))/(sqrt_num_in_points-1.)
            e_grid=np.arange(np.amin(es),np.amax(es)+(e_step/2.),e_step)
        parity=[-1,1]
        roots=[]
        for j in test_grid:
            for k in test_grid:
                for par in parity:
                    x_guess=complex(j,par*k)
                    guesses=[]
                    deltas=[]
                    converged=[]
                    if len(orders)==3:
                        guesses.append([x_guess,second_order_e_1(x_guess,orders[2],orders[1],orders[0],popt)])
                        guesses.append([x_guess,second_order_e_2(x_guess,orders[2],orders[1],orders[0],popt)])
                        deltas.append([complex(0.,0.),complex(0.,0.)])
                        deltas.append([complex(0.,0.),complex(0.,0.)])
                        converged.append(False)
                        converged.append(False)
                    elif len(orders)==4:
                        guesses.append([x_guess,third_order_e_1(x_guess,orders[3],orders[2],orders[1],orders[0],popt)])
                        guesses.append([x_guess,third_order_e_2(x_guess,orders[3],orders[2],orders[1],orders[0],popt)])
                        guesses.append([x_guess,third_order_e_3(x_guess,orders[3],orders[2],orders[1],orders[0],popt)])
                        deltas.append([complex(0.,0.),complex(0.,0.)])
                        deltas.append([complex(0.,0.),complex(0.,0.)])
                        deltas.append([complex(0.,0.),complex(0.,0.)])
                        converged.append(False)
                        converged.append(False)
                        converged.append(False)
                    else:
                        for l in e_grid:
                            e_guess=complex(l,0.)
                            guess_converged=False
                            for m in range(max_nr_iter):
                                old_e_guess=e_guess
                                e_guess=e_guess-(func([x_guess,e_guess],orders,popt)/dfuncde([x_guess,e_guess],orders,popt))
                                if np.abs(e_guess-old_e_guess)<nr_tol:
                                    guess_converged=True
                                    break
                            if guess_converged:
                                guesses.append([x_guess,e_guess])
                                deltas.append([complex(0.,0.),complex(0.,0.)])
                                converged.append(False)
                    guesses=np.array(guesses)
                    deltas=np.array(deltas)
                    for l in range(max_nr_iter):
                        for m in range(len(deltas)):
                            if converged[m]==False:
                                fvec=np.array([func(guesses[m],orders,popt),dfuncdx(guesses[m],orders,popt)])
                                kmat=np.array([[dfuncdx(guesses[m],orders,popt),dfuncde(guesses[m],orders,popt)],[ddfuncddx(guesses[m],orders,popt),ddfuncdxde(guesses[m],orders,popt)]])
                                invkmat=np.linalg.inv(kmat)
                                deltas[m]=-np.dot(invkmat,fvec)
                            else:
                                deltas[m]=np.array([0.,0.])
                        guesses+=deltas
                        for m in range(len(deltas)):
                            if np.sum(np.abs(deltas[m]))<nr_tol:
                                converged[m]=True
                        n=0
                        for m in converged:
                            if m==False:
                                n=1
                                break
                        if n==0:
                            break
                    for l in range(len(guesses)):
                        if converged[l]:
                            not_same=True
                            for m in roots:
                                same_test=np.sum(np.abs(m-guesses[l]))
                                if same_test<diff_sol_tol:
                                    not_same=False
                                    break
                            if not_same:
                                roots.append(guesses[l])
        roots=np.array(roots)
        if verbose>=2.:
            print('Avoided Crossing:')
            print(i)
            print('Roots:')
            print(roots)
            print('')
        self.av_roots[wac-1]=[i,roots]
        return roots
 
if __name__ == "__main__":
    test=GPA()
    test.get_data_from_file(filename='test.txt')
    test.make_stab_plot()
    test.detect_avoided_crossings(each_side=20,plot=False)
    print(test.avoided_crossings)
    test.gpa(wac=1,orders=[2,1,0],nr_tol=1.0e-5,diff_sol_tol=1.0e-2,sqrt_num_in_points=10,max_nr_iter=1000,gd_step=0.01,gd_num_its=100,gd_tol=1.e-5,verbose=4.)
    test.detect_avoided_crossings(each_side=20,plot=True)


