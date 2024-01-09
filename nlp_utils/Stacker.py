

class Stacker():

    def __init__(self):
        pass

    def remove_nones(self, data:list)->list:
        self.data_list = data.copy()
        self.dict_data = {index:item for index,item in enumerate(self.data_list)}
        self.filtered_data = [item for item in self.data_list if item is not None and len(item)>0]
        return self.filtered_data

    def add_nones(self, new_data:list)->list:
        new_dict_data = {}
        
        for item in self.dict_data.items():
            key = item[0]
            value = item[1]
            
            if value is not None and len(value)>0:

                new_dict_data[key] = new_data.pop(0)
            else:
                new_dict_data[key] = None

        return list(new_dict_data.values())