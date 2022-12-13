from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelEncoder


def get_layer_activations(model, arr_X):
    layer_outputs = [layer.output for layer in model.layers] 
    # Extracts the outputs of the top 12 layers
    activation_model = Model(inputs=model.input, outputs=layer_outputs) 
    activations = activation_model.predict(arr_X)
    
    return activations

def get_activations_shape():
    all_layer_activation=[]
    for activation in activations:
        all_layer_activation.append(activation)
        print(activation.shape)
    return all_layer_activation
    

def display_layer_activations(all_layer_activation, x_data):
    names=['input','encoded1','encoded2','encoded3','encoded4', 'fc1','cl']
    layers = [4,5,6]
    for i in layers:
    #     if l in layers:
    #     layer_name=names[l]
        layer_activation=all_layer_activation[i]
        if len(layer_activation.shape)>2:
            x_data = layer_activation.reshape((layer_activation.shape[0], -1))
        else:
            x_data = layer_activation
        standard = StandardScaler()
        x_std = standard.fit_transform(x_data)


        label_encoder = LabelEncoder()
    #     y_tr, y_tt= train_test_split(labels,  test_size=0.20)
        y = label_encoder.fit_transform(predictions)

        tsne = TSNE(n_components=2, random_state=0)  # n_components means you mean to plot your dimensional data to 2D
        x_test_2d = tsne.fit_transform(x_std)

        print()

        markers = ('s', 'd', 'o', '^', 'v', '8', 's', 'p', "_", '2')
        color_map = {0: 'red', 1: 'blue', 2: 'lightgreen', 3: 'purple', 4: 'cyan', 5: 'black', 6: 'yellow', 7: 'magenta',
                 8: 'plum', 9: 'yellowgreen'}
        for idx, cl in enumerate(np.unique(y)):
            class_name=''
            if cl==0: class_name='blush'
            elif cl==1: class_name='cream'
            elif cl==2: class_name='eye'
            elif cl==3: class_name='gloss'
            elif cl==4: class_name='mascara'
            plt.scatter(x=x_test_2d[y == cl, 0], y=x_test_2d[y == cl, 1], c=color_map[idx],
                    label=class_name)
        plt.xlabel('X in t-SNE')
        plt.ylabel('Y in t-SNE')
    #         if l==13:
        plt.legend(loc='best', bbox_to_anchor=(0.5, 0., 0.5, 0.5))
        plt.title('Layer-'+str(i)+ '-'+names[i])#layer_name
        plt.show()