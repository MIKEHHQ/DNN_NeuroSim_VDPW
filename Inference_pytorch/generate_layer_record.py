import numpy as np
import csv
import random

input_precision = 8
weight_precision = 8

f = open('./layer_record_JLP_depthwise/trace_command_JLP_depthwise.sh', "w")
f.write('./main ./NetWork_JLP_depthwise.csv '+str(weight_precision)+' '+str(input_precision)+' ')
with open('NetWork_JLP_depthwise.csv') as csvfile:
    csv_reader = csv.reader(csvfile, delimiter=',')
    line_count = 0
    for row in csv_reader:
        input_height = int(row[0])
        input_width = int(row[1])
        kernel_input_channel = int(row[2])
        kernel_height = int(row[3])
        kernel_width = int(row[4])
        kernel_output_channel = int(row[5])
        stride = int(row[7])
        output_height = int(input_height/stride)
        output_width = int(input_width/stride)
        print(input_height,input_width)
        print(kernel_input_channel,kernel_height,kernel_width,kernel_output_channel)
        input = np.random.randint(2, size=(kernel_input_channel*kernel_height*kernel_width, output_height*output_width*input_precision))
        if kernel_output_channel==0:
            kernel_output_channel = kernel_input_channel
            weight = np.zeros((kernel_input_channel*kernel_height*kernel_width, kernel_output_channel))
            for i in range(kernel_height*kernel_width):
                weight[i*kernel_input_channel:(i+1)*kernel_input_channel, :] = np.diag(np.random.random(size=kernel_input_channel))
        else:            
            weight = np.random.random(size=(kernel_input_channel*kernel_height*kernel_width, kernel_output_channel))
        np.savetxt('./layer_record_JLP_depthwise/input_'+str(line_count)+'.csv', input, delimiter=",", fmt='%s')
        np.savetxt('./layer_record_JLP_depthwise/weight_'+str(line_count)+'.csv', weight, delimiter=",", fmt='%10.5f')
        f.write('./layer_record_JLP_depthwise/weight_'+str(line_count)+'.csv'+' '+'./layer_record_JLP_depthwise/input_'+str(line_count)+'.csv'+' ')

        line_count += 1
        print("done ", line_count)
