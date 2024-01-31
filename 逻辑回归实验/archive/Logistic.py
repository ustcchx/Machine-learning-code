import numpy as np

class LogisticRegression:

    def __init__(self, df,test_df,penalty="l2", gamma=0, fit_intercept=True):
        err_msg = "penalty must be 'l1' or 'l2', but got: {}".format(penalty)
        assert penalty in ["l2", "l1"], err_msg
        self.penalty = penalty
        self.gamma = gamma
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.df=df
        self.beta=np.array([1]*(len(df.columns)))
        self.test_df=test_df

    def loss(self) :
        len_=len(self.df)
        loss=0
        for i in range(len_) :
            temp_arr=np.array(self.df.iloc[i,:-1])
            temp_arr_1=np.append(temp_arr,1)
            loss=loss-self.df['Loan_Status'][i]*np.dot(self.beta,temp_arr_1)+np.log(1+np.exp(np.dot(self.beta,temp_arr_1)))
        if self.fit_intercept==False :
            pass
        elif self.fit_intercept==True and self.penalty=="l1" :
            for i in range(len(self.beta-1)) :
                loss+=self.gamma*abs(self.beta[i])   
        elif self.fit_intercept==True and self.penalty=="l2" :
            sum=0
            for i in range(len(self.beta-1)) :
                sum+=abs(self.beta[i])**2
            loss+=self.gamma*np.sqrt(sum)
        return loss
    
    def sigmoid(self, x):
        x_=np.append(x,1)
        z=np.dot(self.beta,x_)
        if z>0 :
            return 1/(1+np.exp(-z))
        else :
            return np.exp(z)/(1+np.exp(z))
    
    def L_beta_2(self) :
        len_=len(self.df.columns)
        len_index=len(self.df)
        sum_=np.array(len_*[[0]*len_])
        for i in range(len_index) :
            temp_arr=np.array(self.df.iloc[i,:-1])
            temp_arr_1=np.append(temp_arr,1)
            sum_=sum_+self.sigmoid(temp_arr)*(1-self.sigmoid(temp_arr))*np.dot(np.array([temp_arr_1]).T,np.array([temp_arr_1]))
        return sum_
    
    def L_beta_1(self) :
        len_=len(self.df.columns)
        len_index=len(self.df)
        sum_=np.array(len_*[0])
        for i in range(len_index) :
            temp_arr=np.array(self.df.iloc[i,:-1])
            temp_arr_1=np.append(temp_arr,1)
            sum_=sum_-(self.df['Loan_Status'][i]-self.sigmoid(temp_arr))*temp_arr_1
        if self.fit_intercept and self.penalty=="l1":
            for i in range(len(sum_)-1) :
                if self.beta[i]<0 :
                    sum_[i]-=self.gamma
                else :
                    sum_[i]+=self.gamma
                
        elif self.fit_intercept and self.penalty=="l2":
            temp=np.sqrt(np.dot((self.beta)[:-1],(self.beta)[:-1]))
            for i in range(len(sum_)-1) :
                sum_[i]+=self.gamma*(self.beta)[i]/temp
        return sum_
    
    def fit(self, lr=0.01, tol=1e-7, max_iter=1e7):
        self.lr=lr
        self.tol=tol
        self.max_iter=max_iter
        beta_new=np.array([0]*(len(self.df.columns)))
        time=0
        loss_list=[]
        if self.penalty =='l2' :
            while True :
                gradient=self.L_beta_1()
                beta_new=self.beta-self.lr*gradient
                if np.dot(gradient,gradient)<self.tol :
                    break  
                self.beta=beta_new.copy()
                time+=1
                if time>=self.max_iter :
                    break
                loss_list.append(self.loss())
            return loss_list
        else :
            while True :
                gradient=self.L_beta_1()
                beta_new=([self.beta]-self.lr*np.dot(np.array([gradient]),np.linalg.inv(np.array(self.L_beta_2().T)))[0])[0]
                if np.dot(gradient,gradient)<self.tol :
                    break  
                self.beta=beta_new.copy()
                if np.dot(gradient,gradient)<0.003 :
                    self.lr=0.01
                time+=1
                if time>=self.max_iter :
                    break
                loss_list.append(self.loss())
            return loss_list
    
    def predict(self) :
        len_=len(self.test_df)
        predict=[]
        for i in range(len_) :
            temp_arr=np.array(self.test_df.iloc[i,:-1])
            flag=self.sigmoid(temp_arr)
            if flag>=0.5 :
                if (self.test_df)['Loan_Status'][i] == 1 :
                    predict.append([1,1])
                else :
                    predict.append([1,0])
            else :
                if (self.test_df)['Loan_Status'][i] == 1 :
                    predict.append([0,1])
                else :
                    predict.append([0,0])
        self.predict=predict
        return predict
    
    def get_accurate(self) :
        accurate=0
        for i in range(len(self.predict)) :
            if (self.predict)[i][0] == (self.predict)[i][1] :
                accurate+=1
        return accurate/len(self.predict)
    
    def get_R_score(self) :
        TP,FP,FN=0,0,0
        for i in range(len(self.predict)) :
            if (self.predict)[i] == [1,1] :
                TP+=1
            elif (self.predict)[i] == [1,0] :
                FP+=1
            elif (self.predict)[i] == [0,1] :
                FN+=1
        self.R_score=TP/(TP+FN)
        return self.R_score
    
    def get_P_score(self) :
        TP,FP,FN=0,0,0
        for i in range(len(self.predict)) :
            if (self.predict)[i] == [1,1] :
                TP+=1
            elif (self.predict)[i] == [1,0] :
                FP+=1
            elif (self.predict)[i] == [0,1] :
                FN+=1
        self.P_score=TP/(TP+FP)
        return self.P_score
    
    def f1_score(self) :
        R=self.get_R_score()
        P=self.get_P_score()
        self.f1_score=2*P*R/(P+R)
        return self.f1_score


