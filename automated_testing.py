
import os, random
from os import path
import sys
from fsplit.filesplit import Filesplit
from pathlib import Path
import pathlib
import shutil
import subprocess
import nltk
from nltk.featstruct import _substitute_bindings
import pandas as pd
import re
import numpy as np 
import seaborn as sn
import matplotlib.pyplot as plt
from decimal import Decimal, ROUND_HALF_UP


#checks wether the user_path is a path
def user_path(input_path): 
    if not path.exists(input_path):
        print("path does not exist")
    else: 
        return

# takes a random trace from the folder
def select_random_trace(input_path):
    random_trace = random.choice(os.listdir(input_path))
    random_trace_path =  input_path + "/" + random_trace 
    return random_trace_path

#calculate the splitsize (24h / 10 Min = 144)
def splitsize_calculator(random_filepath):
    randomfile_size = os.path.getsize(random_filepath)
    chunksize = randomfile_size / 144
    return int(chunksize)


#the callback function for the filesplitter function
def split_cb():
    print("file has been split")

#Split a flow into ten minute flows (24h / 10 Min = 144)
def filesplitter(file_path, chunksize, output_path):
    fs = Filesplit()
    target_path = str(output_path) + "/splitfiles"
    if not os.path.exists(target_path):
        os.mkdir(target_path) 
    fs.split(file=file_path, split_size=chunksize, output_dir=target_path, callback=split_cb())
    target_path_obsolete_file = str(target_path) + "/fs_manifest.csv"
    #delete the file that lets the splitfile library know on how to reassemble files. 
    os.remove(target_path_obsolete_file)

#selects ten random straces and copies them to a new folder
def random_split_trace_selector():
    current_path = pathlib.Path().absolute()
    splitfiles_path = str(current_path) + "/splitfiles"
    target_path = str(current_path) + "/splitfiles/filter_traces/"
    if not os.path.exists(target_path):
        os.mkdir(target_path) 
    for i in range(0,100):
        random_trace_total_path = select_random_trace(splitfiles_path)
        random_trace_basename = os.path.basename(select_random_trace(splitfiles_path))
        random_trace_new_location = target_path + random_trace_basename
        # I tend to get this unexplicable error: [Errno 21] Is a directory: '/home/moshtart/Dokumente/Arbeit/IOT/our_tests/splitfiles/filter_traces'
        # when copiing some of the traces, the following exception is for debugging purposes. The error occurs irregularily. 
        try:
            shutil.copy(random_trace_total_path, random_trace_new_location)
        except IOError as e:
            print("\n" +"="*70)
            print(e)
            print("traces in " + target_path)
            print("-"*70)
            for  i in os.listdir(target_path):
                print(i)

#deletes the ./filter_traces folder
def folder_deleter(path):
    shutil.rmtree(path)
    #shutil.rmtree("./splitfiles/filter_traces")

def move_a_folder():
    shutil.move("./splitfiles/filter_traces", "./filter_traces") 

def filter_creator():
    os.system("./ALGORITHMEN/mrsh_cuckoo.exe -f ./filter_traces -g")

#creates a mrsh filter for a specific file 
def filter_creator_with_path(path):
    os.system("./ALGORITHMEN/mrsh_cuckoo.exe -f {} -g".format(path))

def tokenizer(string):
    tokens = nltk.word_tokenize(string)
    return tokens

#tokenizes the whole line of a input string
def output_tokenizer(input):
    output_format = ["Trace", "TotalChunks", "ChunksDetected"]
    output_tokenized = tokenizer(str(input))
    output_clean = dict(zip(output_format, output_tokenized))
    return output_clean

#extracts the devicename from the user input
def device_name_parser(input):
    devicename = os.path.basename(os.path.normpath(input))
    return devicename

#builds the first dataframe 
def filter_vs_files_comparison(devicename):
    
    #output_raw = subprocess.getoutput("./ALGORITHMEN/mrsh_cuckoo.exe -s ./filter_traces -c ~/Dokumente/Arbeit/IOT/LSIF/Wans_Cam") #./IOT_TRACES
    command_output = subprocess.run(
        ["./ALGORITHMEN/mrsh_cuckoo.exe", "-s", "./filter_traces", "-c", "IOT_TRACES/"],
        stdout=subprocess.PIPE,
        universal_newlines=True).stdout

    a = []

    output_lines = iter(command_output.splitlines())
    for line in output_lines:
        device = str(output_tokenizer(line).get("Trace"))
        chunks_total = str(output_tokenizer(line).get("TotalChunks"))
        chunks_detected = str(output_tokenizer(line).get("ChunksDetected"))
        chunks_detected_div_chunks_total = int((int(chunks_detected) / int(chunks_total))*100)

        line_np_arrary = [device,chunks_total,chunks_detected, chunks_detected_div_chunks_total]
        a.append(line_np_arrary)
    dataframe_device = pd.DataFrame(a,columns=["Trace", "TotalChunks", devicename,devicename + "_%"])
    return(dataframe_device)
    #return output_clean

def single_filter_vs_files_comparison(devicename, filterpath):
    
    #output_raw = subprocess.getoutput("./ALGORITHMEN/mrsh_cuckoo.exe -s ./filter_traces -c ~/Dokumente/Arbeit/IOT/LSIF/Wans_Cam") #./IOT_TRACES
    command_output = subprocess.run(
        ["./ALGORITHMEN/mrsh_cuckoo.exe", "-s", filterpath, "-c", "IOT_TRACES/"],
        stdout=subprocess.PIPE,
        universal_newlines=True).stdout

    a = []

    output_lines = iter(command_output.splitlines())
    for line in output_lines:
        device = str(output_tokenizer(line).get("Trace"))
        chunks_total = str(output_tokenizer(line).get("TotalChunks"))
        chunks_detected = str(output_tokenizer(line).get("ChunksDetected"))
        chunks_detected_div_chunks_total = Decimal((int(chunks_detected) / int(chunks_total)))
        chunks_detected_div_chunks_total_float = round(chunks_detected_div_chunks_total,4)

        line_np_arrary = [device,chunks_total,chunks_detected, chunks_detected_div_chunks_total_float]
        a.append(line_np_arrary)
    dataframe_device = pd.DataFrame(a,columns=["Trace", "TotalChunks", devicename,devicename + "_%"])
    return(dataframe_device)

def path_expander(rel_path):
    expand_path = os.path.expanduser(rel_path)
    return expand_path

def directory_paths_to_list(dir):
    full_path = path_expander(dir)
    listOfDirectories = [os.path.join(full_path, file) for file in os.listdir(full_path)]
    new_list = []
    for i in listOfDirectories:
        if os.path.isdir(i):
            if not i.endswith(".git"):
                new_list.append(i)
    return new_list


def testrun_for_single_device(path_to_device_traces):

    #in case of a ~ this expands the relative path to the full path
    full_path = str(path_expander(path_to_device_traces))
    
    devicename = device_name_parser(full_path)
    current_path = pathlib.Path().absolute()
    
    #checks wether the given path exists
    user_path(full_path)

    #select a random trace from that the specific device has created
    random_trace = select_random_trace(full_path)

    #calculate the splitsize for this trace
    splitsize = splitsize_calculator(random_trace)

    #split the trace into smaller once of 10min length
    filesplitter(random_trace, splitsize, current_path)

    #select 10 random traces that serve as a filter when comparing against all others and copy them to a new location
    random_split_trace_selector()

    #move this location out of the reach of the mrshcf algorithm otherwise it will be compared against as well
    move_a_folder()

    #create a filter from the 100 traces within the folder
    filter_creator()
 
    #run mrshcf with the 10 traces as a filter and against all other traces
    result = filter_vs_files_comparison(devicename)

    #clean up all the created folders
    folder_deleter("./filter_traces")
    folder_deleter("./splitfiles")
    os.remove("mrsh.sig")

    return result

def testrun_for_single_device_no_split(path_to_device_traces):
    
    #in case of a ~ this expands the relative path to the full path
    full_path = str(path_expander(path_to_device_traces))
    
    devicename = device_name_parser(full_path)
    
    #checks wether the given path exists
    user_path(full_path)

    #select a random trace from that the specific device has created
    random_trace = select_random_trace(full_path)

    #create a filter from the random trace
    filter_creator_with_path(random_trace)
 
    #run mrshcf with the filter and against all other traces
    result = single_filter_vs_files_comparison(devicename, random_trace)

    #clean up all the created folders
    #folder_deleter("./filter_traces")
    #folder_deleter("./splitfiles")
    os.remove("mrsh.sig")

    return result

#takes a list of all devices and does a testrun with each of them
#then it collects the results in a dataframe and saves them in a 
# dict. This dictionary is then turned into a large dataframe     
def raw_matrix_builder_second_dataset(lst_of_device_paths):
    DF_Collection = {}

    #fill the dictionary with the datarframes of every device
    for device in lst_of_device_paths:
        devicename = device_name_parser(device)
        testresults_for_one_device = testrun_for_single_device_no_split(device) 
        DF_Collection[devicename] =  testresults_for_one_device
        #Debug routine
        print("\n" +"="*70)
        print(devicename)
        print("-"*70)
        print(DF_Collection[devicename])

    #now join the dictionary so that you get one big dataframe 
    DF_Collection = { k: v.set_index("Trace") for k, v in DF_Collection.items()}

    
    #concatenate all the dataframes in the dict
    df = pd.concat(DF_Collection.values(),axis=1)

    #remove the duplicate columns by name (here it is TotalChunks)
    df = df.loc[:,~df.columns.duplicated()]


    #turn this larger dataframe into a .csv-file
    df.to_csv("detection_matrix.csv")
    return(df)

#takes a list of all devices and does a testrun with each of them
#then it collects the results in a dataframe and saves them in a 
# dict. This dictionary is then turned into a large dataframe     
def raw_matrix_builder(lst_of_device_paths):
    DF_Collection = {}

    #fill the dictionary with the datarframes of every device
    for device in lst_of_device_paths:
        devicename = device_name_parser(device)
        testresults_for_one_device = testrun_for_single_device(device) 
        DF_Collection[devicename] =  testresults_for_one_device
        #Debug routine
        print("\n" +"="*70)
        print(devicename)
        print("-"*70)
        print(DF_Collection[devicename])

    #now join the dictionary so that you get one big dataframe 
    DF_Collection = { k: v.set_index("Trace") for k, v in DF_Collection.items()}

    
    #concatenate all the dataframes in the dict
    df = pd.concat(DF_Collection.values(),axis=1)

    #remove the duplicate columns by name (here it is TotalChunks)
    df = df.loc[:,~df.columns.duplicated()]


    #turn this larger dataframe into a .csv-file
    df.to_csv("detection_matrix.csv")
    return(df)


def clean_dataframe_names(name):
        # Search for opening bracket in the name followed by
    # any characters reeapted any number of times
    if re.search('\(.*', name):

        # Extract the position of beginning of pattern
        pos = re.search('\(.*', name).start()
  
        # return the cleaned name
        return name[:pos]
  
    else:
        # if clean up needed return the same name
        return name


#take delete the raw chunks_detected colums, calculate the average detection/ confusion for every device 
def matrix_recalculator(dataframe, lst_of_device_paths):
    devicenames = []
    for path in lst_of_device_paths:
        devicenames.append(device_name_parser(path))


    # reset to numeric index
    dataframe = dataframe.reset_index()    

    for device in devicenames:
        dataframe.drop(device, axis=1, inplace=True)
        
        #replace the name of the trace with the name of the respeective device
        dataframe.loc[dataframe['Trace'].str.contains(device, case=False), 'Trace'] = device

    #delete the column TotalChunks it is not needed anymore
    dataframe = dataframe.drop(columns=["TotalChunks"])

    dataframe = dataframe.groupby("Trace").mean()

    #turn this dataframe into a .csv-file
    dataframe.to_csv("recalculated_detection_matrix.csv")


    print(dataframe)

def matrix_visualiser():

    detection_file_df = pd.read_csv("recalculated_detection_matrix_conf.csv")

    devicenames = ["~/Dokumente/Arbeit/IOT/our_tests/captures_IoT-Sentinel/D-LinkDoorSensor", 
                   "~/Dokumente/Arbeit/IOT/our_tests/captures_IoT-Sentinel/D-LinkHomeHub", 
                   "~/Dokumente/Arbeit/IOT/our_tests/captures_IoT-Sentinel/D-LinkSensor",
                   "~/Dokumente/Arbeit/IOT/our_tests/captures_IoT-Sentinel/D-LinkSiren",
                   "~/Dokumente/Arbeit/IOT/our_tests/captures_IoT-Sentinel/D-LinkSwitch",
                   "~/Dokumente/Arbeit/IOT/our_tests/captures_IoT-Sentinel/D-LinkWaterSensor",
                   "~/Dokumente/Arbeit/IOT/our_tests/captures_IoT-Sentinel/HomeMaticPlug"]
    
    #for path in lst_of_device_paths:
    #    devicenames.append(device_name_parser(path))

    detection_file_df.columns = detection_file_df.columns.str.replace("_%", "")

    detection_file_df = detection_file_df.set_index("Trace")    

    detection_file_df = detection_file_df.reindex(sorted(detection_file_df.columns), axis=1)

    #turn this dataframe into a .csv-file
    detection_file_df.to_csv("recalculated_detection_matrix_crosstab_conf.csv")

    ax = sn.heatmap(detection_file_df, annot=True, cmap="Blues", vmin=0, vmax=100)
    ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 13)
    ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 13)

    #set labels for axis
    ax.set_xlabel("DEVICEFILTER", fontsize=13)
    ax.set_ylabel("DEVICETRACES", fontsize=13)

    #set label and axis labels on top
    #ax.xaxis.set_ticks_position('top')

    #ax.xaxis.set_label_position('top')

    plt.show()

#calculates FP TP TN FN as well as Precision, Recall, F1-Score,	Accuracy, Specificity, AUC
def matrix_stats_calculator():
    
    lst_of_device_paths = directory_paths_to_list("~/Dokumente/Arbeit/IOT/LSIF/")

    devicenames = []
    for path in lst_of_device_paths:
        devicenames.append(device_name_parser(path))
     
    df = pd.read_csv("detection_matrix.csv")

    #df=df[df.columns.drop(list(df.filter(regex='_%')))]
    


    for device in devicenames:
        
        #replace the name of the trace with the name of the respeective device
        df.loc[df['Trace'].str.contains(device, case=False), 'Trace'] = device

    df =df.set_index("Trace")  

    #drop all columns that contain chunkamount only leaving those showing relative amounts
    df.drop([col for col in df.columns if not '_%' in col],axis=1,inplace=True)    

    for device in devicenames:
        columnname = device + "_%"

        #sort a rows values in descending order
        sub_table =df.sort_values(by=[columnname], ascending=False)
        
        #get the first 20 entries for that device 
        sub_table_head_raw = sub_table[columnname].nlargest(20, keep='all') 
        sub_table_head = sub_table_head_raw[(sub_table_head_raw > 0)]

        sub_table_tail =  pd.merge(sub_table[columnname],sub_table_head ,how='left', on="Trace")
        #sub_table_tail = ~sub_table[columnname].nlargest(20, keep='all')
        #sub_table_tail =  pd.merge(sub_table,sub_table_head_raw ,how='left')

        

        #TP - number of traces that are from the device
        TP = sub_table_head[device].count()

        

        sub_table.reindex(sub_table.columns, axis=1)

        
        #FP - number of traces that are not from the device
        if device == "Lumiman_SmartPlug":
            sub_table_head = sub_table[columnname].nlargest(9, keep='all') > 0
            sub_table_tail = ~sub_table[columnname].nlargest(20, keep='all')
            FP_df = sub_table_head.drop(device)
            FP = FP_df.count()
            


            #TN - number of traces are not in the top 9 and are not from the device
            try:  
                N_df = sub_table_tail.drop(device)
                TN = N_df.count()
            except:
                #TN_df = sub_table[~sub_table.isin(sub_table_head)]
                TN = "ERROR"#TN_df.count()     
            
        else:            
            FP_df = sub_table_head.drop(device)
            FP = FP_df.count()

            N_df = sub_table_tail.drop(device)
            TN = N_df.count()

        





        #FN - number of falsely "not matched" straces
        try:
            FN = N_df[device].count()
        except: 
            FN = 0    






        #FN - number of traces that belong to the device but are not ranked in the top 20


  
        print(device + "\n")
        print("-"*70) 
        print(" TP: " + str(TP) + "\n")
        #print(sub_table_head)
        print("-"*70)        
        print(" FP: " + str(FP) + "\n")
        #print(FP_df)
        print("-"*70)        
        print(" TN: " + str(TN) + "\n")
        #print(N_df)
        print("-"*70)        
        print(" FN: " + str(FN))
        print("\n" +"="*70)        
        #print(sub_table_head)
        
    
    #print(df)
        



if __name__=="__main__":

    #testrun_for_single_device("~/Dokumente/Arbeit/IOT/LSIF/Wans_Cam/")

    #lst_of_device_paths = directory_paths_to_list("~/Dokumente/Arbeit/IOT/LSIF/")
    #lst_of_device_paths = directory_paths_to_list("~/Dokumente/Arbeit/IOT/our_tests/captures_IoT-Sentinel/")
    
    #for path in lst_of_device_paths:
    #    devicename = device_name_parser(path)
    #    print(devicename)
        #print(path)
    
    #testrun_for_single_device_no_split("~/Dokumente/Arbeit/IOT/our_tests/captures_IoT-Sentinel/Lightify/")

    #raw_data_matrix = raw_matrix_builder_second_dataset(lst_of_device_paths)

    #raw_data_matrix = raw_matrix_builder(lst_of_device_paths)
    
    #recalculated_matrix = matrix_recalculator(raw_data_matrix, lst_of_device_paths)
    
    #matrix_stats_calculator()

    matrix_visualiser()

    '''
    df = pd.DataFrame({'Device':["D-LinkDayCam",
    "D-LinkDoorSensor",  
    "D-LinkHomeHub", 
    "D-LinkCam",   
    "D-LinkSwitch",   
    "D-LinkWaterSensor",    
    "D-LinkSiren",   
    "D-LinkSensor",   
    "HueBridge",   
    "HueSwitch",   
    "SmarterCoffee",   
    "Smarter iKettle2",   
    "TP-LinkPlugHS110",   
    "TP-LinkPlugHS100",   
    "TP-LinkPlugHS100",   
    "EdimaxPlug1101w",    
    "EdimaxPlug1101w",    
    "EdimaxPlug2101w",    
    "Aria", 
    "HomeMaticPlug",  
    "Lightify",
    "Ednetgateway",   
    "MAXGateway",   
    "WeMoLink", 
    "Withings"],
    "Accuracy Cu-IoT": [100,87,50,100,74,77,57,58,100,100,100,100,96,99,99,87,87,97,100,91,95,95,95,97,100]})
    #"Accuracy TLSH":   [100,100,99,100,95,99,99,95,100,100,97,96,99,96,96,95,100,96,100,99,99,100,100]}) 

    sn.set_style(style="darkgrid")
    ax = sn.barplot(y="Device", x='Accuracy', data=df, color="lightblue")
    
    plt.show()
    '''




    


    


