import numpy as np
from keras import backend as K
from keras.layers import Input

import sys
import importlib
sys.path.insert(0, r'modules')
import evaluation
import statistic
import postprocessing
import qas_export
import embeddings
import dataset_collection
import utils
import fuzzing
from dataset import DatasetLuFactory
import tuner
import mapping
import disamb2

# load data
token_filters = '!"#$%&()*+,-./:;<=>?@[\\]^`{|}~\t\n\''
all_words = embeddings.get_words(r'data\glove\glove.6B.100d.txt')
# eliminate unnecessary characters
tokenizer = embeddings.create_tokenizer(all_words, token_filters)
max_words = 20

data_root = r'data\ha'
filenames_with_meta = [
        ('HomeAutomation-6k.tsv', {'fuse': True}),
        ('HomeAutomationMustPass.tsv', {'fuse': True}),
        ('head-10k-speaker-3Mons.tsv', {'fuse': True}),
        ('head-10k-speaker-2Mons-blind.tsv', {'fuse': True}),
        ('speaker-Jan-real-user-110k-SR-joined.tsv', {'fuse': True}),
        ('Speaker-Random-Mar-2018-153k.tsv', {'test_set': True})]
filenames = [x[0] for x in filenames_with_meta]

collection = dataset_collection.load_fuse(
        'HOMEAUTOMATION', data_root, tokenizer, max_words, filenames_with_meta,
         fuse_on_meta_key='fuse',
         fused_meta={'train_set': True, 'label': 'train_fused'})
collection_ha_only = collection.scope_to_domain('HOMEAUTOMATION')

# ? map to general slots
slot_map = {
        'device_name': 'device_id',
        'device_type': 'device_id',
        'provider_name': 'device_id',
        'order_ref': 'quantifier',
        'numerical_increment': 'setting',
        'unit': 'setting'
        }
collection_slot_mapped = mapping.DomainModifier.map_slots(collection, slot_map)
collection_ha_only_slot_mapped = mapping.DomainModifier.map_slots(collection_ha_only, slot_map)


# create fuzzed data
annotator_slots = {'SLOT_DEVICE_ID': 'DEVICE_ID',
                   'SLOT_LOCATION_AREA': 'LOCATION',
                   'SLOT_LOCATION_KEYWORD': 'LOCATION',
                   'SLOT_NUMERICAL_INCREMENT': 'NUMERICAL_INCREMENT',
                   'SLOT_QUANTIFIER_PREFIX': 'QUANTIFIER',
                   'SLOT_QUANTIFIER_STANDALONE': 'DEVICE_ID',
                   'SLOT_SETTING_ADJ': 'SETTING',
                   'SLOT_SETTING_MODE': 'SETTING',
                   'SLOT_SETTING_NUMERIC': 'SETTING',
                   'SLOT_SETTING_TEMP_DOWN': 'SETTING',
                   'SLOT_SETTING_TEMP_UP': 'SETTING',
                   'SLOT_SETTING_STATE': 'SETTING',
                   'SLOT_SETTING_TYPE': 'SETTING_TYPE',
                   'SLOT_QUANTIFIER': 'QUANTIFIER'}
annotator = fuzzing.Annotator.create(r'data\ha\fuzz', 'patterns', annotator_slots)
annotated_df = annotator.generate_dataframe('HOMEAUTOMATION', 60000)
annotated_dataset = DatasetLuFactory.load_dataset(
        annotated_df, 'annotated',
        collection_slot_mapped.encoders, max_words).with_extra_meta({'label': 'fuzzed', 'train_set': True})

other_domain_id = collection_slot_mapped.encoders['domain_label_encoder'].transform(['other'])[0]
train_set = collection_slot_mapped.single_dataset(lambda ds: ds.meta.get('train_set'))
train_set_other_indices = np.nonzero(train_set.arrays['domain'][:, other_domain_id] == 1)[0]

train_set_fuzzed = train_set.sample(train_set_other_indices).merge(annotated_dataset).shuffle()
collection_fuzzed_slot_mapped = collection_slot_mapped.subcollection(lambda ds: not ds.meta.get('train_set'))
collection_fuzzed_slot_mapped.datasets.append(train_set_fuzzed)
collection_ha_only_fuzzed_slot_mapped = collection_fuzzed_slot_mapped.scope_to_domain('HOMEAUTOMATION')

train_set_augmented = train_set.repeat(5).merge(annotated_dataset).shuffle()
collection_augmented_slot_mapped = collection_slot_mapped.subcollection(lambda ds: not ds.meta.get('train_set'))
collection_augmented_slot_mapped.datasets.append(train_set_augmented)
collection_ha_only_augmented_slot_mapped = collection_augmented_slot_mapped.scope_to_domain('HOMEAUTOMATION')



# gather statistic
#desc_collection = collection.describe()
#desc_collection_fuzzed = collection_fuzzed.describe()
#desc_collection_ha_only = collection_ha_only.describe()
#desc_collection_ha_only_fuzzed = collection_ha_only_fuzzed.describe()
#dataframes = dataset_collection.load_dataframes(data_root, filenames)
#desc_slot_values = statistic.get_slot_values('HOMEAUTOMATION', dataframes, filenames)
#desc_dirst_slots = statistic.get_slot_distr('HOMEAUTOMATION', dataframes, filenames)
#desc_dirst_slots_fuzzed = statistic.get_slot_distr('HOMEAUTOMATION', [annotated_df], ['fuzzed'])
#desc_dirst_intents = statistic.get_intent_distr('HOMEAUTOMATION', dataframes, filenames)


# train model
hyperparam_space = {
        'embed_size': [100],
        'embed_dropout': [0.1],
        'gru_size': [[32]],
        'gru_dropout': [0.1],
        'gru_output_dropout': [0.1],
        'dense_dropout': [0.5],
        'slot_dense': [[32, 32]],
        'intent_dense': [[32, 32]],
        'domain_dense': [[32]],
        'optimizer': ['adam'],
        'slot_activation': ['elu'],
        'intent_activation': ['elu'],
        'domain_activation': ['elu'],
        'slot_loss_weight': [6],
        'intent_loss_weight': [6]
            }

# slot model
s_train_set = collection_ha_only_augmented_slot_mapped.single_dataset(lambda ds: ds.meta.get('train_set'))
s_train_collection = collection_ha_only_augmented_slot_mapped.subcollection(lambda ds: not ds.meta.get('train_set'))
s_tuner = tuner.Tuner(hyperparam_space, s_train_set, s_train_collection, out_filename='results.tsv', mode='s', extra_inputs=[])
s_tuner.pick_model()
s_tuner.train(epochs=10)

s_eval_collection = collection_ha_only.subcollection(lambda ds: not ds.meta.get('train_set'))
s_outputs = evaluation.run_model(s_tuner.model, s_eval_collection.datasets,
                                 s_tuner.input_order, s_tuner.output_order)

s_outputs_resolved = postprocessing.postprocess_slots(
        s_train_collection.encoders, s_eval_collection.encoders, s_outputs,
        [ds.arrays['queries'] for ds in s_eval_collection.datasets],
        disamb2.RuleBasedHaResolver())

stats = evaluation.eval_model(s_eval_collection.datasets, s_outputs_resolved, 
                              s_eval_collection.encoders, mode='s')
s_eval = evaluation.stack(stats)

test_set_index = s_eval_collection.single_dataset_index(lambda ds: ds.meta.get('label') == 'HomeAutomationMustPass.tsv')
test_set = s_eval_collection.datasets[test_set_index]
test_set_output = s_outputs_resolved[test_set_index]
report_per_slot = evaluation.build_per_slot_report(test_set, test_set_output, s_eval_collection.encoders)
report_per_query = evaluation.build_per_query_report(test_set, test_set_output, s_eval_collection.encoders, mode='s', stack=True)

statistic.plot_slot_conf_matrix(test_set, test_set_output, s_eval_collection.encoders, vmax='non_diag_max')



# intent model
i_train_set = collection_ha_only_augmented_slot_mapped.single_dataset(lambda ds: ds.meta.get('train_set'))
i_eval_collection = collection_ha_only_augmented_slot_mapped.subcollection(lambda ds: not ds.meta.get('train_set')).downsample(30000)
i_tuner = tuner.Tuner(hyperparam_space, i_train_set, i_eval_collection,
                       out_filename='results.tsv', mode='i')
i_tuner.pick_model()
i_tuner.train(epochs=30)
#di_tuner.load_model(r'out\ha_di_no_luna_intent')

i_eval_collection = collection_ha_only_fuzzed_slot_mapped.subcollection(lambda ds: not ds.meta.get('train_set'))
i_outputs = evaluation.run_model(i_tuner.model, i_eval_collection.datasets,
                                 i_tuner.input_order, i_tuner.output_order)
stats = evaluation.eval_model(i_eval_collection.datasets, i_outputs, 
                              i_eval_collection.encoders, mode='i')
i_eval = evaluation.stack(stats)

test_set_index = i_eval_collection.single_dataset_index(lambda ds: ds.meta.get('label') == 'head-10k-speaker-3Mons.tsv')
test_set = i_eval_collection.datasets[test_set_index]
test_set_output = i_outputs[test_set_index]
report_per_intent = evaluation.build_per_intent_report(test_set, test_set_output, i_eval_collection.encoders)
report_per_query = evaluation.build_per_query_report(test_set, test_set_output, i_eval_collection.encoders, mode='i')

statistic.plot_intent_conf_matrix(test_set, test_set_output, i_eval_collection.encoders, vmax='non_diag_max')




# domain model
d_train_set = collection_slot_mapped.single_dataset(lambda ds: ds.meta.get('train_set'))
d_eval_collection = collection_slot_mapped.subcollection(lambda ds: not ds.meta.get('train_set')).downsample(30000)
d_tuner = tuner.Tuner(hyperparam_space, d_train_set, d_eval_collection,
                       out_filename='results.tsv', mode='d')
d_tuner.pick_model()
d_tuner.train(epochs=10)
#d_tuner.load_model(r'out\ha_di_no_luna_intent')

d_eval_collection = collection_fuzzed_slot_mapped.subcollection(lambda ds: ds.meta.get('label') == 'Speaker-Random-Mar-2018-153k.tsv')
d_outputs = evaluation.run_model(d_tuner.model, d_eval_collection.datasets,
                                 d_tuner.input_order, d_tuner.output_order)
stats = evaluation.eval_model(d_eval_collection.datasets, d_outputs, 
                              d_eval_collection.encoders, mode='d')
d_eval = evaluation.stack(stats)

test_set_index = d_eval_collection.single_dataset_index(lambda ds: ds.meta.get('label') == 'Speaker-Random-Mar-2018-153k.tsv')
test_set = d_eval_collection.datasets[test_set_index]
test_set_output = d_outputs[test_set_index]
report_per_query = evaluation.build_per_query_report(test_set, test_set_output, d_eval_collection.encoders, mode='d')




# evaluate full model
dis_eval_collection = collection.subcollection(lambda ds: ds.meta.get('label') != 'train-fused')

d_tuner.load_model(r'out\ha_dis_model_augmented\d')
d_outputs = evaluation.run_model(d_tuner.model, dis_eval_collection.datasets,
                                 d_tuner.input_order, d_tuner.output_order)

i_tuner.load_model(r'out\ha_dis_model_augmented\i')
i_outputs = evaluation.run_model(i_tuner.model, dis_eval_collection.datasets,
                                 i_tuner.input_order, i_tuner.output_order)
s_tuner.load_model(r'out\ha_dis_model_augmented\s')
s_outputs = evaluation.run_model(s_tuner.model, dis_eval_collection.datasets,
                                 s_tuner.input_order, s_tuner.output_order)

dis_outputs = [utils.merge_dicts([d, i, s]) for d,i,s in zip(d_outputs, i_outputs, s_outputs)]

dis_outputs_resolved = postprocessing.postprocess_slots(
        collection_slot_mapped.encoders, dis_eval_collection.encoders, dis_outputs,
        [ds.arrays['queries'] for ds in dis_eval_collection.datasets],
        disamb2.RuleBasedHaResolver())
dis_outputs_postprocessed = postprocessing.emulate_intent_slot_after_d_threshold(dis_outputs_resolved, dis_eval_collection.encoders)

stats = evaluation.eval_model(dis_eval_collection.datasets, dis_outputs_postprocessed, 
                              dis_eval_collection.encoders, mode='dis')
dis_eval = evaluation.stack(stats)

test_set_index = dis_eval_collection.single_dataset_index(lambda ds: ds.meta.get('label') == 'HomeAutomationMustPass.tsv')
test_set = dis_eval_collection.datasets[test_set_index]
test_set_output = dis_outputs_postprocessed[test_set_index]
report_per_slot = evaluation.build_per_slot_report(test_set, test_set_output, dis_eval_collection.encoders)
report_per_intent = evaluation.build_per_intent_report(test_set, test_set_output, dis_eval_collection.encoders)
report_per_query = evaluation.build_per_query_report(test_set, test_set_output, dis_eval_collection.encoders, mode='dis', stack=True)

statistic.plot_slot_conf_matrix(test_set, test_set_output, collection.encoders, vmax='non_diag_max')
statistic.plot_intent_conf_matrix(test_set, test_set_output, collection.encoders, vmax='non_diag_max')



# export the model
s_tuner = tuner.Tuner(hyperparam_space, s_train_set, s_eval_collection, mode='s')
s_tuner.load_model(r'out\ha_s_fuzzed_30k')
s_tuner.finalize_model()
s_tuner.export_model(r'out\model')

qas_export.build_input_vocabulary(r'out\model', r'qas\mlg\retail\amd64\app\MLGTools')
qas_export.generate_pipelines(r'out\model', 'ha_tf')
qas_export.finalize_qas_model(r'out\model', 'ha_tf')





