from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier 
from sklearn.svm import SVC


class DecisionTree():
    def __init__(self) -> None:
        self.model = DecisionTreeClassifier()
    
    def fit(self,x,y):
        self.model = self.model.fit(x,y)
    
    def acc(self,x,y):
        return self.model.score(x,y)
    
    def pred(self,x):
        return self.model.predict(x)
    


class RandomForest(DecisionTree):
    def __init__(self) -> None:
        super().__init__()
        self.model = RandomForestClassifier()


class SVM(DecisionTree):
    def __init__(self) -> None:
        super().__init__()
        self.model = SVC()


