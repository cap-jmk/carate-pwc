class Test():
    def __init__(self): 
        self.var1 = 1
        self.var2 = 2
    
    def test_func(self, a=5, b=4): 
        print(locals())
    def return_attributes(self): 
        return vars(self)
if __name__ == "__main__": 
    test = Test()
    print(test.return_attributes())
    print(test.test_func())