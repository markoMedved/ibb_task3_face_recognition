def classification_accuracy(genuines, impostors, start_treshold, end_treshold, step):
    classification_acc = []
    prec = []
    recall = []
    f1 = []

    for treshold in range(start_treshold, end_treshold, step):
        correctly_classified_gen = 0
        correctly_classified_imp = 0
        #count number of correctly classified genuines
        for el in genuines:
            if el <= treshold:
                correctly_classified_gen+=1
        #count number of correctly classified impostors
        for el in impostors:
            if el > treshold:
                correctly_classified_imp+=1
        
        classification_acc.append((correctly_classified_gen + correctly_classified_imp)/(len(genuines) + len(impostors)))
        prec.append((correctly_classified_gen)/(correctly_classified_gen + len(impostors) - correctly_classified_imp))
        recall.append(correctly_classified_gen/len(genuines))
        f1.append((2*prec[-1]*recall[-1])/(prec[-1] + recall[-1] + 0.00000001))
    
    
    indeks = f1.index(max(f1))
    best_treshold = start_treshold + indeks*step

    return best_treshold, classification_acc[indeks], prec[indeks], recall[indeks], f1[indeks]


def plot_curve(genuine, impostors):
    genuine_frequency= []
    genuine_count = []


    impostors_frequency= []
    impostors_count = []

    #create frequency and count
    for i in range(0, int(max(genuine))+1):
        genuine_frequency.append(genuine.count(i))
        genuine_count.append(i)

        #same for impostors
    for i in range(0, int(max(impostors))+1):
        impostors_frequency.append(impostors.count(i))
        impostors_count.append(i)

    plt.plot(genuine_count, genuine_frequency,label="genuines")
    plt.plot(impostors_count, impostors_frequency, label="impostors")

    plt.legend()
    plt.show()


def get_genuines_impostors_scores(dataset,function,n_points=8, radius=1, grid_size=(4,4), scaleFactor = 1.3, minNeighbors=3, minSize=(20,20),useBoundingBox=True):
    #extract feature vectors for all the samples
    genuines = []
    impostors = []
    feature_vectors,image_ids = extract_all_feature_vectors(dataset,function,n_points=n_points, radius=radius, grid_size=grid_size, scaleFactor = scaleFactor, minNeighbors=minNeighbors, minSize=minSize,useBoundingBox=useBoundingBox)
    for feature_vector, image_id in zip(feature_vectors, image_ids):
        #get the identity of the sample
        identity = dataset[dataset["idx"] == image_id]["identity"].values[0]
        #get all the samples of the same identity
        df_gen = dataset[dataset["identity"] == identity]
        #get all the samples of different identity
        df_imp = dataset[dataset["identity"] != identity]
        
        for i in range(len(df_gen)):
            #compute the chi2 distance between the feature vectors
            genuines.append(chi2_distance(feature_vector, feature_vectors[i]))
            
        for i in range(len(df_imp)):
            #compute the chi2 distance between the feature vectors
            impostors.append(chi2_distance(feature_vector, feature_vectors[i]))
    return genuines, impostors