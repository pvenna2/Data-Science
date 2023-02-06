
import csv

#this function is to clean data from text file
def dataCleaning():
    #opening text file in reading mode
    file = open("temperatures.txt" , 'r')
    #data structre to store data from file in cleaned form
    rawLst = []
    for i in file:
        #read line from file and split on base of comma
        #extracting two values for month and year and two value for crossponding temperaturs
        lst = i.split(',')
        year = lst[0][1:5]
        month = lst[0][5:]
        val = lst[1][1:-1]
        rawLst.append([year,month,val])
        year = lst[2][2:6]
        month = lst[2][6:]
        val = lst[3][1:-2]
        #storing both values in lst of cleaned data
        rawLst.append([year , month , val])
    #closing file opened
    file.close()
    #returning cleaned data from file in lst format 
    return rawLst

        

#function to split data in two half
def dataSplit(rawLst):
    #create a list of first 1000 entries
    lst1 = rawLst[:1000]
    #creating list of remaining entries
    lst2 = rawLst[1000:]
    #returing two list which are halved
    return lst1 , lst2

#function to map data 
def mapper(rawLst):

    #creating a dictionary structure
    dataDict = {}
    for i in rawLst:
        #itterating in raw data lst
        year = int(i[0])
        month = int(i[1])
        val = int(i[2])
        #checking if year is in data dictionary already 
        if year in dataDict:
            #append value to consective month
            if(dataDict[year][month-1]<val):
                dataDict[year][month-1] = val
        else:
            #otherwise create key of year and assign month value
            dataDict[year] = [0,0,0,0,0,0,0,0,0,0,0,0]
            dataDict[year][month-1] = val
            

    #returing key pair value data
    return dataDict


#function to sort data dictionary 
def sortFunction(dataDic):
    #it takes the sorted keys by using sort function 
    lstSorted = sorted(dataDic)
    newDic = {}
    #in new dictionary the sorted values are appending with their values
    for i in lstSorted:
        newDic[i] = dataDic[i]
    #return sorted dictionary 
    return newDic


#function to divide data 
def partitionFun(dataDic):
    dic1 = {}
    dic2 = {}
    #created two dictionaries
    for i in dataDic:
        if(i<2016):        
            #for ddata from 2016 to 2020 set it in dic1
            dic1[i] = dataDic[i]
        else:
            #otherwise for data 2010 to 2015 assign it in dic2
            dic2[i] = dataDic[i]
    #return two dictionaries
    return dic1 , dic2
    
#function to find maximum temperature for each year
def reducer(dataDic):
    outLst = []
    for i in dataDic:
        #it takes max temp from each year and append it in lst
        maxTemp = max(dataDic[i])
        outLst.append([i , maxTemp])
    #returning maximum temperature list
    return outLst


#main function 
def main():

    #calling raw data cleaning function 
    rawData = dataCleaning()
    #spliting data in two list
    lst1 , lst2  = dataSplit(rawData)
    #mapping data for each list
    dic1 = mapper(rawData)
    dic2 = mapper(lst2)
    #merging data in one dictionary
    #dic1.update(dic2)

    #sorting data dictionary
    sortedDic = sortFunction(dic1)

    #calling partition data function 
    dic1,dic2 = partitionFun(sortedDic)
    #applying reducer function on each dictionary
    lst1 = reducer(dic1)
    lst2 = reducer(dic2)
    #opening file to write
    file = open("output.csv" , 'w' , newline='')
    with file:
        writer =  csv.writer(file)
        #writing header in file
        writer.writerows([["year" , "Max Temp"]])
        #writing list values for max temperature for each year in file
        writer.writerows(lst1)
        writer.writerows(lst2)
    #closing file
    file.close()
    #writing message to console
    print("OUTPUT FILE GENERATED!")
    

main()