#--coding:utf-8--
from numpy import *
from os import listdir
import operator
def createdataset():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels = ['A','A','B','B']
    return group,labels

def classify0(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX,(dataSetSize,1)) - dataSet
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis = 1)
    distances = sqDistances**0.5
    sortedDistIndicies = distances.argsort()
    classCount = {}
    for i in range(k):
        voteIlabel = labels[sortedDistIndicies[i]]
        classCount[voteIlabel] = classCount.get(voteIlabel,0) + 1
    sortedClassCount = sorted(classCount.iteritems(),
                              key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]
def file2matrix(filename):                         
    fr = open('C:\Python27\datingdataset2.txt')
    #打开文件，按行读入
    arrayOLines = fr.readlines()    
    #获得文件行数 
    numberOfLines = len(arrayOLines)  
    #创建m行n列的零矩阵 
    returnMat = zeros((numberOfLines,3))          
    classLabelVector = []
    index = 0
    for line in arrayOLines:
        line = line.strip()
        #删除行前面的空格
        listFromLine = line.split('\t')
         #根据分隔符划分
        returnMat[index,:] = listFromLine[0:3]
        #取得每一行的内容存起来
        classLabelVector.append(int(listFromLine[-1]))
        index += 1
    return returnMat,classLabelVector
def autonorm(dataSet):
    minvals = dataSet.min(0)
    maxvals = dataSet.max(0)
    ranges = maxvals - minvals
    normdataset = zeros(shape(dataSet))
    m = dataSet.shape[0]
    normdataset = dataSet - tile(minvals,(m,1))
    normdataset = normdataset/tile(ranges,(m,1))
    return normdataset,ranges, minvals
def datingclasstest():
    horatio = 0.10
    datingdatamat,datinglabels = file2matrix('c:\python27\datingtestset')
    normmat, ranges, minvals = autonorm(datingdatamat)
    m = normmat.shape[0]
    numtestvecs = int(m*horatio)
    errorcount = 0.0
    for i in range(numtestvecs):
        classifierresult = classify0(normmat[i,1],normmat[numtestvecs:m,:],\
            datinglabels[numtestvecs:m],3)
        print "the classifier came back with:%d,the real answer is:%d"\
            %(classifierresult,datinglabels[i])
        if (classifierresult !=datinglabels[i]): errorcount += 1.0
    print "the total error rate is:%f" % (errorcount/float(numtestvecs))
def classifyperson():
    resultlist = ['not at all','in small doses','in large doses']
    precenttats = float(raw_input(\
        "perscentage of time spent playing video games?"))
    ffmiles = float(raw_input("frequent filer miles earned per year"))
    icecream = float(raw_input("liters of icecream sonsumed per year"))
    datingdatamat,datinglabels = file2matrix('c:\python27\datingtestset.txt')
    normmat, ranges, minvals = autonorm(datingdatamat)
    inarr = array([ffmiles,precenttats,icecream])
    classifierresult = classify0((inarr-\
        minvals)/ranges,normmat,datinglabels,3)
    print"you will probably like this person:",\
    resultlist[classifierresult - 1]
def img2vector(filename):
    returnvect = zeros((1,1024))
    fr = open('c:\\python27\\0_13.txt')
    for i in range(32):
        linestr = fr.readline()
        for j in range(32):
            returnvect[0,32*i+j] = int(linestr[j])
        return returnvect
def handwritingclasstest():
    hwlabels = []
    trainingfilelist = listdir('c:\\python27\\trainingdigits')
    m = len(trainingfilelist)
    trainningmat = zeros((m,1024))
    for i in range(m):
        filenamestr = trainingfilelist[i]
        filestr = filenamestr.split('_')[0]
        classnumstr = int(filestr.split('_')[0])
        hwlabels.append(classnumstr)
        trainningmat[i,:] = img2vector('trainingdigits/%s' % filenamestr)
    testfilelist = listdir('c:\\python27\\testdigits')
    errorcount = 0.0
    mtest = len(testfilelist)
    for i in range(mtest):
        filenamestr = testfilelist[i]
        filestr = filenamestr.split('_')[0]
        classnumstr = int(filestr.split('_')[0])
        vectorundertest = img2vector('c:\\python27\\testdigits/%s' % filenamestr)
        classifierresult = classify0(vectorundertest, \
            trainningmat,hwlabels,3)
        print "the classifier came back with: %d,the real answer is: %d"\
            % (classifierresult,classnumstr)
        if (classifierresult !=classnumstr):errorcount +=1.0
        print "\n the total number of errors is: %d" % errorcount
        print "\n the total error rate is: %f" % (errorcount/float(mtest))
        