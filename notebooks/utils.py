def get_make_from_title(make_list, title):
    title = title.split(' ')
    for i in range(len(title)):
        if (' '.join(title[0:i+1]) in make_list):
            return ' '.join(title[0:i+1])
    return 'unknwon'

def make_category_vector(cat_list, x):
    vector = [0] * len(cat_list)
    for i, cat in enumerate(cat_list):
        if cat in x:
            vector[i] = 1
    return vector