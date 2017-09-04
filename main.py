
"""
The main program of Incisor Segmentation

"""
#%% 
import argparse

import task1
import task2
import task3



#%% 

def get_args():
    
    parser = argparse.ArgumentParser(description="""A program for model-based segmentation of \
                                     the upper and lower incisors in panoramic radiographs""")
    
    parser.add_argument('-init','--init_method', help="The method of finding initial estimate",
                        action='store', choices=['auto', 'manual'], default="auto")
    
    parser.add_argument('-k', '--k', help="No. of pixels on either side of a model point for grey level model",
                        action='store', type=int, default=10)
    parser.add_argument('-m', '--m', help="No. of sample points on either side of current point for search",
                        action='store', type=int, default=15)
    parser.add_argument('-s', '--skip_amf', help="Skip adaptive median filter in Preprocessing", \
                        action='store_false')
    
    return parser.parse_args()

#%% 
def main():
    
    # Initialisation
    args = get_args()
    k = args.k
    m = args.m
    skip_amf = args.skip_amf   # Adaptive Median Filter can be skipped for faster implementation. 
    auto_estimate = True if args.init_method == 'auto' else False
    
    task2.preprocess_radiographs(skip_amf)    
    
    incisor_list = range(1,9)
    
    # leave-one-out
    for test_img_idx in range(1,15):
        print("Test Image - %d of 14" %(test_img_idx))      
        print("")
        
        asm_list = task1.buildASM(incisor_list, test_img_idx, k) 
 
        final_fit_list = task3.fit_model(asm_list, incisor_list, test_img_idx, m,\
                                         auto_estimate=auto_estimate, save=True) 
                
        task3.evaluate_results(test_img_idx, incisor_list, final_fit_list)
        
if __name__ == '__main__':
    main()
