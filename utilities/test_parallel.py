import multiprocessing
import cv2
import csv
import pdb

def try_image(img_file):
    result = []
    try:
        im = cv2.imread(img_file)
        height, width = im.shape[:2]
        if height > 100 and width > 100:
            result = img_file
    except:
        print("{} does not refer to a valid img file".format(img_file))
    return result


if __name__ == "__main__":
    all_files = []
    all_anns = []
    infiles = ['small_pa_sample.txt','./infiles/pa_2014_juncadd_val.txt'] #'./infiles/pa_2014_ann_train.txt' #
    for file in infiles:
        train_phase = True
        imgs_to_validate = {}
        with open(file,'r') as f:
            reader = csv.reader(f,delimiter='\t')
            for row in reader:
                #print(row[0])
                fname = row[0]
                if(train_phase):
                    anns = list(map(int,row[1:8]))
                    imgs_to_validate[fname] = anns
                else:
                    imgs_to_validate[fname] = []

        imgs_validated = []

        pool = multiprocessing.Pool(processes = 8)
        imgs_validated = pool.map(try_image, imgs_to_validate.keys())
        imgs_validated = [x for x in imgs_validated if x !=[]]  #remove empty lists indicated image was not valid
        cleaned = dict((k, imgs_to_validate[k]) for k in imgs_validated)

        keys_list = list(cleaned.keys())
        values_list = list(cleaned.values())
        all_files.extend(keys_list)
        all_anns.extend(values_list)

    print(all_files)
    print(all_anns)
    pdb.set_trace()
