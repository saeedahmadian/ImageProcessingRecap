	??/??KW@??/??KW@!??/??KW@	?????L???????L??!?????L??"{
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails:??/??KW@W???x??AḌ?8W@Y??T?????rEagerKernelExecute 0*	?rh??0a@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???wӥ?!??6?>@)_???:T??1\??@s;@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice/ܹ0ҋ??!??H??2@)/ܹ0ҋ??1??H??2@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zipy?P?????!<"hA??R@)-??o????1?h#9?@0@:Preprocessing2U
Iterator::Model::ParallelMapV2?(?ޕ?!??*?C/@)?(?ޕ?1??*?C/@:Preprocessing2F
Iterator::Model? ?m?8??!w_?Fu8@))?k{?%??1*???I?!@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate???N???!`????:@)Z??????12??F?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor??
???s?!??̿?^@)??
???s?1??̿?^@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap@??????!?.???J<@)IIC?c?1???@????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.1% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9?????L??Iڄ????X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	W???x??W???x??!W???x??      ??!       "      ??!       *      ??!       2	Ḍ?8W@Ḍ?8W@!Ḍ?8W@:      ??!       B      ??!       J	??T???????T?????!??T?????R      ??!       Z	??T???????T?????!??T?????b      ??!       JCPU_ONLYY?????L??b qڄ????X@