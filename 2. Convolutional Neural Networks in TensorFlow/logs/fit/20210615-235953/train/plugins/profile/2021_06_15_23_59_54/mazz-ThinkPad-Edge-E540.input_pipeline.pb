	??4}?!@??4}?!@!??4}?!@	Q?$?	???Q?$?	???!Q?$?	???"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??4}?!@m?Yg|??A?e?^!@Y??P??C??*	
ףp=?`@2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap6[y?????!???s?\G@)sI?v??1=?~PȐD@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?c?=	l??!~?βH6@)?a??h???1?w??;2@:Preprocessing2U
Iterator::Model::ParallelMapV2???m3??!???}*@)???m3??1???}*@:Preprocessing2F
Iterator::Model??A?V???!??????9@)3???yS??1"???a)@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::TensorSlice)A?G?~?!???Q`@))A?G?~?1???Q`@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?:?G??!???
?R@)gF?N?{?1??1M7@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor)H4?"v?!???6@))H4?"v?1???6@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
device?Your program is NOT input-bound because only 0.7% of the total step time sampled is waiting for input. Therefore, you should focus on reducing other time.no*no9Q?$?	???Ic???+?X@Zno#You may skip the rest of this page.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
	m?Yg|??m?Yg|??!m?Yg|??      ??!       "      ??!       *      ??!       2	?e?^!@?e?^!@!?e?^!@:      ??!       B      ??!       J	??P??C????P??C??!??P??C??R      ??!       Z	??P??C????P??C??!??P??C??b      ??!       JCPU_ONLYYQ?$?	???b qc???+?X@