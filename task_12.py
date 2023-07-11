import pandas as pd
import math


def add_label_vector(df, label_list):
  for x in range(len(label_list)):
    if not math.isnan(label_list[x]):
      df['label_vector'][x]= int(label_list[x])


df = pd.read_json('jobQ3_BOTH_train.json', orient='split')
df_full_ann = pd.read_excel("JobQ3b_full_annotations.xlsx","Data to Annotate")

labels_ann = list(df_full_ann.drop(["Message Id", "Message"], axis=1).columns)

for x in range(len(labels_ann)):
    df_full_ann[labels_ann[x]].replace({1:(x)},inplace=True)

listofzeros = [0] * 2455
df_full_ann['label_vector']=listofzeros

for f in range(len(labels_ann)):
    list_label = list(df_full_ann[labels_ann[f]])
    add_label_vector(df_full_ann, list_label)


label_json = ['coming_home_from_work', 'complaining_about_work','getting_cut_in_hours','getting_fired',
              'getting_hiredjob_seeking','getting_promotedraised', 'going_to_work','losing_job_some_other_way',
              'none_of_the_above_but_jobrelated', 'not_jobrelated','offering_support',  'quitting_a_job']

df_full_ann["label"] = [0] * 2455

final_label = list(df_full_ann['label_vector'])

for x in range(len(final_label)):
    df_full_ann['label'][x]= label_json[final_label[x]]

id_list = []
for x in range(len(list(df_full_ann['Message Id']))):
    id_list.append(int(list(df_full_ann['Message Id'])[x]))

df_concatenate = pd.DataFrame()

df_concatenate['worker_id']=[0] * 2455
df_concatenate['label']=df_full_ann['label']
df_concatenate['message']= df_full_ann['Message']
df_concatenate['message_id']=id_list
df_concatenate['platform']=[0] * 2455
df_concatenate['label_vector']=df_full_ann['label_vector']

frames = [df, df_concatenate]

enhanced_df = pd.concat(frames)

compression_opts = dict(method='zip',archive_name='enhanced_data.csv')
enhanced_df.to_csv('enhanced_data.zip', index=False, compression=compression_opts)


