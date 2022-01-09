import os

class DirHandler(object):
    '''
    A Utility to create a structure as :
                            Train:  Class1
                                    Class2...
                            Test:   Class1
                                    Class2...
                            ...
    path  : Path Where you want to create above structure
    '''
    def __init__(self,path):
        self.path = path

    def make_directory(self, directory):
            if not os.path.exists(directory):
                os.makedirs(directory)
                print(directory, ": Directory has been Created")
                return directory
            else:
                print("Directory Already exists")
                return directory

    def createset(self,mode):
        directory = self.make_directory(os.path.join(self.path, mode))
        gestures = ['zero','one','two']
        for i in gestures:
            self.make_directory(os.path.join(directory, i))
        print("Required Directories Created!")

    def datacount(self,mode):
        datastats = {}
        for files in os.listdir(os.path.join(self.path,mode)):
            try:
                k = (len(os.listdir(os.path.join(self.path,mode,files))))
                datastats[files] = k
            except NotADirectoryError:
                pass
        return {mode:datastats}

    def removeunwanted(self):
        for root,dir,files in  os.walk(os.path.join(self.path)):
            for f in (files):
                if f.endswith('.png') or f.endswith('.jpg') or f.endswith('.jpeg'):
                    pass
                else:
                    os.remove(os.path.join(root,f))


# D = DirHandler("/Users/prathameshsardeshmukh/PycharmProjects/Motor_AI_Test/Dataset")
# D.createset(mode='dataset')
# D.createset(mode='test')
# D.createset(mode='validation')


# print(D.datacount('dataset'))
# D.removeunwanted()