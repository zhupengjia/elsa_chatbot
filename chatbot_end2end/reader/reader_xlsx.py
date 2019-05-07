#!/usr/bin/env python
import os, pandas, re

'''
    Author: Pengjia Zhu (zhupengjia@gmail.com)
'''

class ReaderXLSX:
    '''
        dialog reader from xlsx file

        Input:
            - dialog_file: xlsx file of rule definition

    '''

    def __init__(self, dialog_file):
        self.dialog_file = dialog_file
        self._read()

    def _read(self):
        if not os.path.exists(self.dialog_file):
            raise('{} not exists!!!'.format(self.dialog_file))
        
        self.dialogs = self._parse_dialog()
        self.entities = self._parse_entity()
        self.actions = self._parse_action()

    def _parse_dialog(self):
        dialogs = pandas.read_excel(self.dialog_file, sheet_name="dialogs", index_col='id')
        dialogs['child_id'] = dialogs['child_id'].apply(self.__decode_childID)
        dialogs["needed_entity"]  = dialogs["needed_entity"].apply(lambda x: [s.strip() for s in re.split("[,| ]", x) if len(s.strip())>0] if isinstance(x, str) else None)
        dialogs["unneeded_entity"]  = dialogs["unneeded_entity"].apply(lambda x: [s.strip() for s in re.split("[,| ]", x) if len(s.strip())>0] if isinstance(x, str) else None)
        dialogs["user_says"]  = dialogs["user_says"].apply(lambda x: [s.strip() for s in re.split("\n", x) if len(s.strip())>0] if isinstance(x, str) else None)
        dialogs["response"]  = dialogs["response"].apply(lambda x: [s.strip() for s in re.split("\n", x) if len(s.strip())>0] if isinstance(x, str) else None)
        dialogs["action"] = dialogs["action"].apply(lambda x: x.strip() if isinstance(x, str) and len(x.strip()) > 0 else None)
        return dialogs

    def _parse_entity(self):
        entities = {"keywords":None, "regex":None, "ner":None, "ner_name_replace":None}
        entities_table = pandas.read_excel(self.dialog_file, sheet_name="entities").iloc[:, 1:]
        if entities_table.shape[0] < 3:
            return entities
        for i in range(entities_table.shape[1]):
            name = entities_table.columns[i]
            name_replace = entities_table.iloc[0, i]
            entity_type = entities_table.iloc[1, i]
            values = [str(x).strip() for x in entities_table.iloc[2:, i].dropna().tolist() if len(str(x).strip())>0]
            if entity_type is None or name is None:
                continue
            name = name.strip()
            entity_type = entity_type.strip().lower()
            if name_replace is not None:
                if entities["ner_name_replace"] is None: entities["ner_name_replace"] = {}
                entities["ner_name_replace"][name] = name_replace
            if entity_type == "ner":
                if entities["ner"] is None: entities["ner"] = []
                entities["ner"].append(name)
            else:
                if len(values) < 1:
                    continue
                if entity_type == "keywords":
                    if entities["keywords"] is None: entities["keywords"] = {}
                    entities["keywords"][name] = values
                elif entity_type == "regex":
                    if entities["regex"] is None: entities["regex"] = {}
                    entities["regex"][name] = values[0]
        return entities

    def _parse_action(self):
        action_table = pandas.read_excel(self.dialog_file, sheet_name="actions", header=None).dropna()
        if action_table.shape[0] < 1:
            return {}
        actions = {}
        for i in range(action_table.shape[0]):
            name = action_table.iloc[i, 0].strip()
            action = ["    "+x for x in re.split("\n", action_table.iloc[i, 1])]
            action.insert(0, "def {}(entities):".format(name))
            action = "\n".join(action)
            exec(action)
            actions[name] = eval(name)
        return actions
   
    def __decode_childID(self, IDs):
        '''
            convert childID to list
        '''
        if isinstance(IDs, str):
            IDs2 = []
            for i in re.split('[,，]', IDs):
                if i.isdigit():
                    IDs2.append(int(i))
                else:
                    itmp = [int(x) for x in re.split('[~-]', i) if len(x.strip())>0]
                    if len(itmp) > 1:
                        IDs2 += range(itmp[0], itmp[1]+1)
                    else:
                        IDs2.append(int(itmp[0]))
            return IDs2
        elif isinstance(IDs, int):
            return [IDs]
        elif isinstance(IDs, list):
            if isinstance(IDs[0], int):
                return IDs
            elif isinstance(IDs[0], str):
                return [int(x) for x in IDs]
        return None

if __name__ == "__main__":
    r = ReaderXLSX("../../data/babi/babi.xlsx")
    print(r.actions)
    print(r.actions["getrestinfo"]({}))
