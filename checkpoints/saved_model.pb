??

??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 ":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.5.02v2.5.0-rc3-213-ga4dfb8d1a718??
?
conv1_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameconv1_layer/kernel
?
&conv1_layer/kernel/Read/ReadVariableOpReadVariableOpconv1_layer/kernel*&
_output_shapes
: *
dtype0
x
conv1_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *!
shared_nameconv1_layer/bias
q
$conv1_layer/bias/Read/ReadVariableOpReadVariableOpconv1_layer/bias*
_output_shapes
: *
dtype0
?
conv2_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*#
shared_nameconv2_layer/kernel
?
&conv2_layer/kernel/Read/ReadVariableOpReadVariableOpconv2_layer/kernel*&
_output_shapes
: @*
dtype0
x
conv2_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv2_layer/bias
q
$conv2_layer/bias/Read/ReadVariableOpReadVariableOpconv2_layer/bias*
_output_shapes
:@*
dtype0
?
conv3_layer/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@*#
shared_nameconv3_layer/kernel
?
&conv3_layer/kernel/Read/ReadVariableOpReadVariableOpconv3_layer/kernel*&
_output_shapes
:@@*
dtype0
x
conv3_layer/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*!
shared_nameconv3_layer/bias
q
$conv3_layer/bias/Read/ReadVariableOpReadVariableOpconv3_layer/bias*
_output_shapes
:@*
dtype0
y
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? @*
shared_namedense_1/kernel
r
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel*
_output_shapes
:	? @*
dtype0
p
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*
shared_namedense_1/bias
i
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes
:@*
dtype0
v
output/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@
*
shared_nameoutput/kernel
o
!output/kernel/Read/ReadVariableOpReadVariableOpoutput/kernel*
_output_shapes

:@
*
dtype0
n
output/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_nameoutput/bias
g
output/bias/Read/ReadVariableOpReadVariableOpoutput/bias*
_output_shapes
:
*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
b
total_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_1
[
total_1/Read/ReadVariableOpReadVariableOptotal_1*
_output_shapes
: *
dtype0
b
count_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_1
[
count_1/Read/ReadVariableOpReadVariableOpcount_1*
_output_shapes
: *
dtype0
b
total_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	total_2
[
total_2/Read/ReadVariableOpReadVariableOptotal_2*
_output_shapes
: *
dtype0
b
count_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name	count_2
[
count_2/Read/ReadVariableOpReadVariableOpcount_2*
_output_shapes
: *
dtype0
?
Adam/conv1_layer/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameAdam/conv1_layer/kernel/m
?
-Adam/conv1_layer/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv1_layer/kernel/m*&
_output_shapes
: *
dtype0
?
Adam/conv1_layer/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv1_layer/bias/m

+Adam/conv1_layer/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv1_layer/bias/m*
_output_shapes
: *
dtype0
?
Adam/conv2_layer/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @**
shared_nameAdam/conv2_layer/kernel/m
?
-Adam/conv2_layer/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv2_layer/kernel/m*&
_output_shapes
: @*
dtype0
?
Adam/conv2_layer/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv2_layer/bias/m

+Adam/conv2_layer/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv2_layer/bias/m*
_output_shapes
:@*
dtype0
?
Adam/conv3_layer/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@**
shared_nameAdam/conv3_layer/kernel/m
?
-Adam/conv3_layer/kernel/m/Read/ReadVariableOpReadVariableOpAdam/conv3_layer/kernel/m*&
_output_shapes
:@@*
dtype0
?
Adam/conv3_layer/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv3_layer/bias/m

+Adam/conv3_layer/bias/m/Read/ReadVariableOpReadVariableOpAdam/conv3_layer/bias/m*
_output_shapes
:@*
dtype0
?
Adam/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? @*&
shared_nameAdam/dense_1/kernel/m
?
)Adam/dense_1/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/m*
_output_shapes
:	? @*
dtype0
~
Adam/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/dense_1/bias/m
w
'Adam/dense_1/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/m*
_output_shapes
:@*
dtype0
?
Adam/output/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@
*%
shared_nameAdam/output/kernel/m
}
(Adam/output/kernel/m/Read/ReadVariableOpReadVariableOpAdam/output/kernel/m*
_output_shapes

:@
*
dtype0
|
Adam/output/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*#
shared_nameAdam/output/bias/m
u
&Adam/output/bias/m/Read/ReadVariableOpReadVariableOpAdam/output/bias/m*
_output_shapes
:
*
dtype0
?
Adam/conv1_layer/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: **
shared_nameAdam/conv1_layer/kernel/v
?
-Adam/conv1_layer/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv1_layer/kernel/v*&
_output_shapes
: *
dtype0
?
Adam/conv1_layer/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *(
shared_nameAdam/conv1_layer/bias/v

+Adam/conv1_layer/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv1_layer/bias/v*
_output_shapes
: *
dtype0
?
Adam/conv2_layer/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @**
shared_nameAdam/conv2_layer/kernel/v
?
-Adam/conv2_layer/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv2_layer/kernel/v*&
_output_shapes
: @*
dtype0
?
Adam/conv2_layer/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv2_layer/bias/v

+Adam/conv2_layer/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv2_layer/bias/v*
_output_shapes
:@*
dtype0
?
Adam/conv3_layer/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@@**
shared_nameAdam/conv3_layer/kernel/v
?
-Adam/conv3_layer/kernel/v/Read/ReadVariableOpReadVariableOpAdam/conv3_layer/kernel/v*&
_output_shapes
:@@*
dtype0
?
Adam/conv3_layer/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*(
shared_nameAdam/conv3_layer/bias/v

+Adam/conv3_layer/bias/v/Read/ReadVariableOpReadVariableOpAdam/conv3_layer/bias/v*
_output_shapes
:@*
dtype0
?
Adam/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	? @*&
shared_nameAdam/dense_1/kernel/v
?
)Adam/dense_1/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/kernel/v*
_output_shapes
:	? @*
dtype0
~
Adam/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*$
shared_nameAdam/dense_1/bias/v
w
'Adam/dense_1/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1/bias/v*
_output_shapes
:@*
dtype0
?
Adam/output/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:@
*%
shared_nameAdam/output/kernel/v
}
(Adam/output/kernel/v/Read/ReadVariableOpReadVariableOpAdam/output/kernel/v*
_output_shapes

:@
*
dtype0
|
Adam/output/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*#
shared_nameAdam/output/bias/v
u
&Adam/output/bias/v/Read/ReadVariableOpReadVariableOpAdam/output/bias/v*
_output_shapes
:
*
dtype0

NoOpNoOp
?@
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*??
value??B?? B??
?
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
 
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
R
trainable_variables
	variables
regularization_losses
	keras_api
h

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
R
 trainable_variables
!	variables
"regularization_losses
#	keras_api
h

$kernel
%bias
&trainable_variables
'	variables
(regularization_losses
)	keras_api
R
*trainable_variables
+	variables
,regularization_losses
-	keras_api
h

.kernel
/bias
0trainable_variables
1	variables
2regularization_losses
3	keras_api
h

4kernel
5bias
6trainable_variables
7	variables
8regularization_losses
9	keras_api
?
:iter

;beta_1

<beta_2
	=decay
>learning_ratem}m~mm?$m?%m?.m?/m?4m?5m?v?v?v?v?$v?%v?.v?/v?4v?5v?
F
0
1
2
3
$4
%5
.6
/7
48
59
 
F
0
1
2
3
$4
%5
.6
/7
48
59
?
?layer_regularization_losses
trainable_variables
@non_trainable_variables
Alayer_metrics

Blayers
regularization_losses
Cmetrics
	variables
 
^\
VARIABLE_VALUEconv1_layer/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEconv1_layer/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
Dlayer_regularization_losses
trainable_variables
Enon_trainable_variables
Flayer_metrics

Glayers
	variables
Hmetrics
regularization_losses
 
 
 
?
Ilayer_regularization_losses
trainable_variables
Jnon_trainable_variables
Klayer_metrics

Llayers
	variables
Mmetrics
regularization_losses
^\
VARIABLE_VALUEconv2_layer/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEconv2_layer/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1

0
1
 
?
Nlayer_regularization_losses
trainable_variables
Onon_trainable_variables
Player_metrics

Qlayers
	variables
Rmetrics
regularization_losses
 
 
 
?
Slayer_regularization_losses
 trainable_variables
Tnon_trainable_variables
Ulayer_metrics

Vlayers
!	variables
Wmetrics
"regularization_losses
^\
VARIABLE_VALUEconv3_layer/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEconv3_layer/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

$0
%1

$0
%1
 
?
Xlayer_regularization_losses
&trainable_variables
Ynon_trainable_variables
Zlayer_metrics

[layers
'	variables
\metrics
(regularization_losses
 
 
 
?
]layer_regularization_losses
*trainable_variables
^non_trainable_variables
_layer_metrics

`layers
+	variables
ametrics
,regularization_losses
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

.0
/1

.0
/1
 
?
blayer_regularization_losses
0trainable_variables
cnon_trainable_variables
dlayer_metrics

elayers
1	variables
fmetrics
2regularization_losses
YW
VARIABLE_VALUEoutput/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEoutput/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE

40
51

40
51
 
?
glayer_regularization_losses
6trainable_variables
hnon_trainable_variables
ilayer_metrics

jlayers
7	variables
kmetrics
8regularization_losses
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
 
 
 
?
0
1
2
3
4
5
6
7
	8

l0
m1
n2
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
4
	ototal
	pcount
q	variables
r	keras_api
D
	stotal
	tcount
u
_fn_kwargs
v	variables
w	keras_api
D
	xtotal
	ycount
z
_fn_kwargs
{	variables
|	keras_api
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

o0
p1

q	variables
QO
VARIABLE_VALUEtotal_14keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_14keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUE
 

s0
t1

v	variables
QO
VARIABLE_VALUEtotal_24keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUE
QO
VARIABLE_VALUEcount_24keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUE
 

x0
y1

{	variables
?
VARIABLE_VALUEAdam/conv1_layer/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/conv1_layer/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/conv2_layer/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/conv2_layer/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/conv3_layer/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/conv3_layer/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/mRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/mPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output/kernel/mRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/output/bias/mPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/conv1_layer/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/conv1_layer/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/conv2_layer/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/conv2_layer/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
VARIABLE_VALUEAdam/conv3_layer/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/conv3_layer/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
}{
VARIABLE_VALUEAdam/dense_1/kernel/vRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUEAdam/dense_1/bias/vPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
|z
VARIABLE_VALUEAdam/output/kernel/vRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
xv
VARIABLE_VALUEAdam/output/bias/vPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*/
_output_shapes
:?????????  *
dtype0*$
shape:?????????  
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv1_layer/kernelconv1_layer/biasconv2_layer/kernelconv2_layer/biasconv3_layer/kernelconv3_layer/biasdense_1/kerneldense_1/biasoutput/kerneloutput/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *+
f&R$
"__inference_signature_wrapper_4472
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename&conv1_layer/kernel/Read/ReadVariableOp$conv1_layer/bias/Read/ReadVariableOp&conv2_layer/kernel/Read/ReadVariableOp$conv2_layer/bias/Read/ReadVariableOp&conv3_layer/kernel/Read/ReadVariableOp$conv3_layer/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOp!output/kernel/Read/ReadVariableOpoutput/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOptotal_1/Read/ReadVariableOpcount_1/Read/ReadVariableOptotal_2/Read/ReadVariableOpcount_2/Read/ReadVariableOp-Adam/conv1_layer/kernel/m/Read/ReadVariableOp+Adam/conv1_layer/bias/m/Read/ReadVariableOp-Adam/conv2_layer/kernel/m/Read/ReadVariableOp+Adam/conv2_layer/bias/m/Read/ReadVariableOp-Adam/conv3_layer/kernel/m/Read/ReadVariableOp+Adam/conv3_layer/bias/m/Read/ReadVariableOp)Adam/dense_1/kernel/m/Read/ReadVariableOp'Adam/dense_1/bias/m/Read/ReadVariableOp(Adam/output/kernel/m/Read/ReadVariableOp&Adam/output/bias/m/Read/ReadVariableOp-Adam/conv1_layer/kernel/v/Read/ReadVariableOp+Adam/conv1_layer/bias/v/Read/ReadVariableOp-Adam/conv2_layer/kernel/v/Read/ReadVariableOp+Adam/conv2_layer/bias/v/Read/ReadVariableOp-Adam/conv3_layer/kernel/v/Read/ReadVariableOp+Adam/conv3_layer/bias/v/Read/ReadVariableOp)Adam/dense_1/kernel/v/Read/ReadVariableOp'Adam/dense_1/bias/v/Read/ReadVariableOp(Adam/output/kernel/v/Read/ReadVariableOp&Adam/output/bias/v/Read/ReadVariableOpConst*6
Tin/
-2+	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *&
f!R
__inference__traced_save_4862
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv1_layer/kernelconv1_layer/biasconv2_layer/kernelconv2_layer/biasconv3_layer/kernelconv3_layer/biasdense_1/kerneldense_1/biasoutput/kerneloutput/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcounttotal_1count_1total_2count_2Adam/conv1_layer/kernel/mAdam/conv1_layer/bias/mAdam/conv2_layer/kernel/mAdam/conv2_layer/bias/mAdam/conv3_layer/kernel/mAdam/conv3_layer/bias/mAdam/dense_1/kernel/mAdam/dense_1/bias/mAdam/output/kernel/mAdam/output/bias/mAdam/conv1_layer/kernel/vAdam/conv1_layer/bias/vAdam/conv2_layer/kernel/vAdam/conv2_layer/bias/vAdam/conv3_layer/kernel/vAdam/conv3_layer/bias/vAdam/dense_1/kernel/vAdam/dense_1/bias/vAdam/output/kernel/vAdam/output/bias/v*5
Tin.
,2**
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *)
f$R"
 __inference__traced_restore_4995??
?
?
E__inference_conv1_layer_layer_call_and_return_conditional_losses_4105

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????   2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?

?
$__inference_model_layer_call_fn_4375
input_1!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@
	unknown_5:	? @
	unknown_6:@
	unknown_7:@

	unknown_8:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_43272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????  : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????  
!
_user_specified_name	input_1
?&
?
?__inference_model_layer_call_and_return_conditional_losses_4439
input_1*
conv1_layer_4410: 
conv1_layer_4412: *
conv2_layer_4416: @
conv2_layer_4418:@*
conv3_layer_4422:@@
conv3_layer_4424:@
dense_1_4428:	? @
dense_1_4430:@
output_4433:@

output_4435:

identity??#conv1_layer/StatefulPartitionedCall?#conv2_layer/StatefulPartitionedCall?#conv3_layer/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?output/StatefulPartitionedCall?
#conv1_layer/StatefulPartitionedCallStatefulPartitionedCallinput_1conv1_layer_4410conv1_layer_4412*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv1_layer_layer_call_and_return_conditional_losses_41052%
#conv1_layer/StatefulPartitionedCall?
max_pooling_1/PartitionedCallPartitionedCall,conv1_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_max_pooling_1_layer_call_and_return_conditional_losses_40692
max_pooling_1/PartitionedCall?
#conv2_layer/StatefulPartitionedCallStatefulPartitionedCall&max_pooling_1/PartitionedCall:output:0conv2_layer_4416conv2_layer_4418*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2_layer_layer_call_and_return_conditional_losses_41232%
#conv2_layer/StatefulPartitionedCall?
max_pooling_2/PartitionedCallPartitionedCall,conv2_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_max_pooling_2_layer_call_and_return_conditional_losses_40812
max_pooling_2/PartitionedCall?
#conv3_layer/StatefulPartitionedCallStatefulPartitionedCall&max_pooling_2/PartitionedCall:output:0conv3_layer_4422conv3_layer_4424*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv3_layer_layer_call_and_return_conditional_losses_41412%
#conv3_layer/StatefulPartitionedCall?
flatt_layer/PartitionedCallPartitionedCall,conv3_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatt_layer_layer_call_and_return_conditional_losses_41532
flatt_layer/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall$flatt_layer/PartitionedCall:output:0dense_1_4428dense_1_4430*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_41662!
dense_1/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0output_4433output_4435*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_output_layer_call_and_return_conditional_losses_41822 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0$^conv1_layer/StatefulPartitionedCall$^conv2_layer/StatefulPartitionedCall$^conv3_layer/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????  : : : : : : : : : : 2J
#conv1_layer/StatefulPartitionedCall#conv1_layer/StatefulPartitionedCall2J
#conv2_layer/StatefulPartitionedCall#conv2_layer/StatefulPartitionedCall2J
#conv3_layer/StatefulPartitionedCall#conv3_layer/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:X T
/
_output_shapes
:?????????  
!
_user_specified_name	input_1
?&
?
?__inference_model_layer_call_and_return_conditional_losses_4189

inputs*
conv1_layer_4106: 
conv1_layer_4108: *
conv2_layer_4124: @
conv2_layer_4126:@*
conv3_layer_4142:@@
conv3_layer_4144:@
dense_1_4167:	? @
dense_1_4169:@
output_4183:@

output_4185:

identity??#conv1_layer/StatefulPartitionedCall?#conv2_layer/StatefulPartitionedCall?#conv3_layer/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?output/StatefulPartitionedCall?
#conv1_layer/StatefulPartitionedCallStatefulPartitionedCallinputsconv1_layer_4106conv1_layer_4108*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv1_layer_layer_call_and_return_conditional_losses_41052%
#conv1_layer/StatefulPartitionedCall?
max_pooling_1/PartitionedCallPartitionedCall,conv1_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_max_pooling_1_layer_call_and_return_conditional_losses_40692
max_pooling_1/PartitionedCall?
#conv2_layer/StatefulPartitionedCallStatefulPartitionedCall&max_pooling_1/PartitionedCall:output:0conv2_layer_4124conv2_layer_4126*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2_layer_layer_call_and_return_conditional_losses_41232%
#conv2_layer/StatefulPartitionedCall?
max_pooling_2/PartitionedCallPartitionedCall,conv2_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_max_pooling_2_layer_call_and_return_conditional_losses_40812
max_pooling_2/PartitionedCall?
#conv3_layer/StatefulPartitionedCallStatefulPartitionedCall&max_pooling_2/PartitionedCall:output:0conv3_layer_4142conv3_layer_4144*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv3_layer_layer_call_and_return_conditional_losses_41412%
#conv3_layer/StatefulPartitionedCall?
flatt_layer/PartitionedCallPartitionedCall,conv3_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatt_layer_layer_call_and_return_conditional_losses_41532
flatt_layer/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall$flatt_layer/PartitionedCall:output:0dense_1_4167dense_1_4169*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_41662!
dense_1/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0output_4183output_4185*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_output_layer_call_and_return_conditional_losses_41822 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0$^conv1_layer/StatefulPartitionedCall$^conv2_layer/StatefulPartitionedCall$^conv3_layer/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????  : : : : : : : : : : 2J
#conv1_layer/StatefulPartitionedCall#conv1_layer/StatefulPartitionedCall2J
#conv2_layer/StatefulPartitionedCall#conv2_layer/StatefulPartitionedCall2J
#conv3_layer/StatefulPartitionedCall#conv3_layer/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
H
,__inference_max_pooling_1_layer_call_fn_4075

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_max_pooling_1_layer_call_and_return_conditional_losses_40692
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
a
E__inference_flatt_layer_layer_call_and_return_conditional_losses_4153

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:?????????? 2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
*__inference_conv1_layer_layer_call_fn_4626

inputs!
unknown: 
	unknown_0: 
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv1_layer_layer_call_and_return_conditional_losses_41052
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?

?
$__inference_model_layer_call_fn_4212
input_1!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@
	unknown_5:	? @
	unknown_6:@
	unknown_7:@

	unknown_8:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_41892
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????  : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????  
!
_user_specified_name	input_1
?8
?
?__inference_model_layer_call_and_return_conditional_losses_4514

inputsD
*conv1_layer_conv2d_readvariableop_resource: 9
+conv1_layer_biasadd_readvariableop_resource: D
*conv2_layer_conv2d_readvariableop_resource: @9
+conv2_layer_biasadd_readvariableop_resource:@D
*conv3_layer_conv2d_readvariableop_resource:@@9
+conv3_layer_biasadd_readvariableop_resource:@9
&dense_1_matmul_readvariableop_resource:	? @5
'dense_1_biasadd_readvariableop_resource:@7
%output_matmul_readvariableop_resource:@
4
&output_biasadd_readvariableop_resource:

identity??"conv1_layer/BiasAdd/ReadVariableOp?!conv1_layer/Conv2D/ReadVariableOp?"conv2_layer/BiasAdd/ReadVariableOp?!conv2_layer/Conv2D/ReadVariableOp?"conv3_layer/BiasAdd/ReadVariableOp?!conv3_layer/Conv2D/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?output/BiasAdd/ReadVariableOp?output/MatMul/ReadVariableOp?
!conv1_layer/Conv2D/ReadVariableOpReadVariableOp*conv1_layer_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02#
!conv1_layer/Conv2D/ReadVariableOp?
conv1_layer/Conv2DConv2Dinputs)conv1_layer/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
conv1_layer/Conv2D?
"conv1_layer/BiasAdd/ReadVariableOpReadVariableOp+conv1_layer_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"conv1_layer/BiasAdd/ReadVariableOp?
conv1_layer/BiasAddBiasAddconv1_layer/Conv2D:output:0*conv1_layer/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2
conv1_layer/BiasAdd?
conv1_layer/ReluReluconv1_layer/BiasAdd:output:0*
T0*/
_output_shapes
:?????????   2
conv1_layer/Relu?
max_pooling_1/MaxPoolMaxPoolconv1_layer/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling_1/MaxPool?
!conv2_layer/Conv2D/ReadVariableOpReadVariableOp*conv2_layer_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02#
!conv2_layer/Conv2D/ReadVariableOp?
conv2_layer/Conv2DConv2Dmax_pooling_1/MaxPool:output:0)conv2_layer/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2_layer/Conv2D?
"conv2_layer/BiasAdd/ReadVariableOpReadVariableOp+conv2_layer_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02$
"conv2_layer/BiasAdd/ReadVariableOp?
conv2_layer/BiasAddBiasAddconv2_layer/Conv2D:output:0*conv2_layer/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2_layer/BiasAdd?
conv2_layer/ReluReluconv2_layer/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2_layer/Relu?
max_pooling_2/MaxPoolMaxPoolconv2_layer/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling_2/MaxPool?
!conv3_layer/Conv2D/ReadVariableOpReadVariableOp*conv3_layer_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02#
!conv3_layer/Conv2D/ReadVariableOp?
conv3_layer/Conv2DConv2Dmax_pooling_2/MaxPool:output:0)conv3_layer/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv3_layer/Conv2D?
"conv3_layer/BiasAdd/ReadVariableOpReadVariableOp+conv3_layer_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02$
"conv3_layer/BiasAdd/ReadVariableOp?
conv3_layer/BiasAddBiasAddconv3_layer/Conv2D:output:0*conv3_layer/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv3_layer/BiasAdd?
conv3_layer/ReluReluconv3_layer/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv3_layer/Reluw
flatt_layer/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatt_layer/Const?
flatt_layer/ReshapeReshapeconv3_layer/Relu:activations:0flatt_layer/Const:output:0*
T0*(
_output_shapes
:?????????? 2
flatt_layer/Reshape?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	? @*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMulflatt_layer/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_1/Relu?
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:@
*
dtype02
output/MatMul/ReadVariableOp?
output/MatMulMatMuldense_1/Relu:activations:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
output/MatMul?
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
output/BiasAdd/ReadVariableOp?
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
output/BiasAdd?
IdentityIdentityoutput/BiasAdd:output:0#^conv1_layer/BiasAdd/ReadVariableOp"^conv1_layer/Conv2D/ReadVariableOp#^conv2_layer/BiasAdd/ReadVariableOp"^conv2_layer/Conv2D/ReadVariableOp#^conv3_layer/BiasAdd/ReadVariableOp"^conv3_layer/Conv2D/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????  : : : : : : : : : : 2H
"conv1_layer/BiasAdd/ReadVariableOp"conv1_layer/BiasAdd/ReadVariableOp2F
!conv1_layer/Conv2D/ReadVariableOp!conv1_layer/Conv2D/ReadVariableOp2H
"conv2_layer/BiasAdd/ReadVariableOp"conv2_layer/BiasAdd/ReadVariableOp2F
!conv2_layer/Conv2D/ReadVariableOp!conv2_layer/Conv2D/ReadVariableOp2H
"conv3_layer/BiasAdd/ReadVariableOp"conv3_layer/BiasAdd/ReadVariableOp2F
!conv3_layer/Conv2D/ReadVariableOp!conv3_layer/Conv2D/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?

?
A__inference_dense_1_layer_call_and_return_conditional_losses_4166

inputs1
matmul_readvariableop_resource:	? @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	? @*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
?
*__inference_conv2_layer_layer_call_fn_4646

inputs!
unknown: @
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2_layer_layer_call_and_return_conditional_losses_41232
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?	
?
@__inference_output_layer_call_and_return_conditional_losses_4707

inputs0
matmul_readvariableop_resource:@
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
*__inference_conv3_layer_layer_call_fn_4666

inputs!
unknown:@@
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv3_layer_layer_call_and_return_conditional_losses_41412
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?	
?
"__inference_signature_wrapper_4472
input_1!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@
	unknown_5:	? @
	unknown_6:@
	unknown_7:@

	unknown_8:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *(
f#R!
__inference__wrapped_model_40632
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????  : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????  
!
_user_specified_name	input_1
?V
?
__inference__traced_save_4862
file_prefix1
-savev2_conv1_layer_kernel_read_readvariableop/
+savev2_conv1_layer_bias_read_readvariableop1
-savev2_conv2_layer_kernel_read_readvariableop/
+savev2_conv2_layer_bias_read_readvariableop1
-savev2_conv3_layer_kernel_read_readvariableop/
+savev2_conv3_layer_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop,
(savev2_output_kernel_read_readvariableop*
&savev2_output_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop&
"savev2_total_1_read_readvariableop&
"savev2_count_1_read_readvariableop&
"savev2_total_2_read_readvariableop&
"savev2_count_2_read_readvariableop8
4savev2_adam_conv1_layer_kernel_m_read_readvariableop6
2savev2_adam_conv1_layer_bias_m_read_readvariableop8
4savev2_adam_conv2_layer_kernel_m_read_readvariableop6
2savev2_adam_conv2_layer_bias_m_read_readvariableop8
4savev2_adam_conv3_layer_kernel_m_read_readvariableop6
2savev2_adam_conv3_layer_bias_m_read_readvariableop4
0savev2_adam_dense_1_kernel_m_read_readvariableop2
.savev2_adam_dense_1_bias_m_read_readvariableop3
/savev2_adam_output_kernel_m_read_readvariableop1
-savev2_adam_output_bias_m_read_readvariableop8
4savev2_adam_conv1_layer_kernel_v_read_readvariableop6
2savev2_adam_conv1_layer_bias_v_read_readvariableop8
4savev2_adam_conv2_layer_kernel_v_read_readvariableop6
2savev2_adam_conv2_layer_bias_v_read_readvariableop8
4savev2_adam_conv3_layer_kernel_v_read_readvariableop6
2savev2_adam_conv3_layer_bias_v_read_readvariableop4
0savev2_adam_dense_1_kernel_v_read_readvariableop2
.savev2_adam_dense_1_bias_v_read_readvariableop3
/savev2_adam_output_kernel_v_read_readvariableop1
-savev2_adam_output_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:**
dtype0*?
value?B?*B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:**
dtype0*g
value^B\*B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0-savev2_conv1_layer_kernel_read_readvariableop+savev2_conv1_layer_bias_read_readvariableop-savev2_conv2_layer_kernel_read_readvariableop+savev2_conv2_layer_bias_read_readvariableop-savev2_conv3_layer_kernel_read_readvariableop+savev2_conv3_layer_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableop(savev2_output_kernel_read_readvariableop&savev2_output_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop"savev2_total_1_read_readvariableop"savev2_count_1_read_readvariableop"savev2_total_2_read_readvariableop"savev2_count_2_read_readvariableop4savev2_adam_conv1_layer_kernel_m_read_readvariableop2savev2_adam_conv1_layer_bias_m_read_readvariableop4savev2_adam_conv2_layer_kernel_m_read_readvariableop2savev2_adam_conv2_layer_bias_m_read_readvariableop4savev2_adam_conv3_layer_kernel_m_read_readvariableop2savev2_adam_conv3_layer_bias_m_read_readvariableop0savev2_adam_dense_1_kernel_m_read_readvariableop.savev2_adam_dense_1_bias_m_read_readvariableop/savev2_adam_output_kernel_m_read_readvariableop-savev2_adam_output_bias_m_read_readvariableop4savev2_adam_conv1_layer_kernel_v_read_readvariableop2savev2_adam_conv1_layer_bias_v_read_readvariableop4savev2_adam_conv2_layer_kernel_v_read_readvariableop2savev2_adam_conv2_layer_bias_v_read_readvariableop4savev2_adam_conv3_layer_kernel_v_read_readvariableop2savev2_adam_conv3_layer_bias_v_read_readvariableop0savev2_adam_dense_1_kernel_v_read_readvariableop.savev2_adam_dense_1_bias_v_read_readvariableop/savev2_adam_output_kernel_v_read_readvariableop-savev2_adam_output_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *8
dtypes.
,2*	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : @:@:@@:@:	? @:@:@
:
: : : : : : : : : : : : : : @:@:@@:@:	? @:@:@
:
: : : @:@:@@:@:	? @:@:@
:
: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:%!

_output_shapes
:	? @: 

_output_shapes
:@:$	 

_output_shapes

:@
: 


_output_shapes
:
:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:,(
&
_output_shapes
:@@: 

_output_shapes
:@:%!

_output_shapes
:	? @: 

_output_shapes
:@:$ 

_output_shapes

:@
: 

_output_shapes
:
:, (
&
_output_shapes
: : !

_output_shapes
: :,"(
&
_output_shapes
: @: #

_output_shapes
:@:,$(
&
_output_shapes
:@@: %

_output_shapes
:@:%&!

_output_shapes
:	? @: '

_output_shapes
:@:$( 

_output_shapes

:@
: )

_output_shapes
:
:*

_output_shapes
: 
?	
?
@__inference_output_layer_call_and_return_conditional_losses_4182

inputs0
matmul_readvariableop_resource:@
-
biasadd_readvariableop_resource:

identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:@
*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
E__inference_conv2_layer_layer_call_and_return_conditional_losses_4123

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?
?
%__inference_output_layer_call_fn_4716

inputs
unknown:@

	unknown_0:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_output_layer_call_and_return_conditional_losses_41822
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????@: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
A__inference_dense_1_layer_call_and_return_conditional_losses_4688

inputs1
matmul_readvariableop_resource:	? @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?MatMul/ReadVariableOp?
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes
:	? @*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
MatMul?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2	
BiasAddX
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:P L
(
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?&
?
?__inference_model_layer_call_and_return_conditional_losses_4407
input_1*
conv1_layer_4378: 
conv1_layer_4380: *
conv2_layer_4384: @
conv2_layer_4386:@*
conv3_layer_4390:@@
conv3_layer_4392:@
dense_1_4396:	? @
dense_1_4398:@
output_4401:@

output_4403:

identity??#conv1_layer/StatefulPartitionedCall?#conv2_layer/StatefulPartitionedCall?#conv3_layer/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?output/StatefulPartitionedCall?
#conv1_layer/StatefulPartitionedCallStatefulPartitionedCallinput_1conv1_layer_4378conv1_layer_4380*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv1_layer_layer_call_and_return_conditional_losses_41052%
#conv1_layer/StatefulPartitionedCall?
max_pooling_1/PartitionedCallPartitionedCall,conv1_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_max_pooling_1_layer_call_and_return_conditional_losses_40692
max_pooling_1/PartitionedCall?
#conv2_layer/StatefulPartitionedCallStatefulPartitionedCall&max_pooling_1/PartitionedCall:output:0conv2_layer_4384conv2_layer_4386*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2_layer_layer_call_and_return_conditional_losses_41232%
#conv2_layer/StatefulPartitionedCall?
max_pooling_2/PartitionedCallPartitionedCall,conv2_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_max_pooling_2_layer_call_and_return_conditional_losses_40812
max_pooling_2/PartitionedCall?
#conv3_layer/StatefulPartitionedCallStatefulPartitionedCall&max_pooling_2/PartitionedCall:output:0conv3_layer_4390conv3_layer_4392*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv3_layer_layer_call_and_return_conditional_losses_41412%
#conv3_layer/StatefulPartitionedCall?
flatt_layer/PartitionedCallPartitionedCall,conv3_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatt_layer_layer_call_and_return_conditional_losses_41532
flatt_layer/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall$flatt_layer/PartitionedCall:output:0dense_1_4396dense_1_4398*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_41662!
dense_1/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0output_4401output_4403*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_output_layer_call_and_return_conditional_losses_41822 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0$^conv1_layer/StatefulPartitionedCall$^conv2_layer/StatefulPartitionedCall$^conv3_layer/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????  : : : : : : : : : : 2J
#conv1_layer/StatefulPartitionedCall#conv1_layer/StatefulPartitionedCall2J
#conv2_layer/StatefulPartitionedCall#conv2_layer/StatefulPartitionedCall2J
#conv3_layer/StatefulPartitionedCall#conv3_layer/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:X T
/
_output_shapes
:?????????  
!
_user_specified_name	input_1
??
?
 __inference__traced_restore_4995
file_prefix=
#assignvariableop_conv1_layer_kernel: 1
#assignvariableop_1_conv1_layer_bias: ?
%assignvariableop_2_conv2_layer_kernel: @1
#assignvariableop_3_conv2_layer_bias:@?
%assignvariableop_4_conv3_layer_kernel:@@1
#assignvariableop_5_conv3_layer_bias:@4
!assignvariableop_6_dense_1_kernel:	? @-
assignvariableop_7_dense_1_bias:@2
 assignvariableop_8_output_kernel:@
,
assignvariableop_9_output_bias:
'
assignvariableop_10_adam_iter:	 )
assignvariableop_11_adam_beta_1: )
assignvariableop_12_adam_beta_2: (
assignvariableop_13_adam_decay: 0
&assignvariableop_14_adam_learning_rate: #
assignvariableop_15_total: #
assignvariableop_16_count: %
assignvariableop_17_total_1: %
assignvariableop_18_count_1: %
assignvariableop_19_total_2: %
assignvariableop_20_count_2: G
-assignvariableop_21_adam_conv1_layer_kernel_m: 9
+assignvariableop_22_adam_conv1_layer_bias_m: G
-assignvariableop_23_adam_conv2_layer_kernel_m: @9
+assignvariableop_24_adam_conv2_layer_bias_m:@G
-assignvariableop_25_adam_conv3_layer_kernel_m:@@9
+assignvariableop_26_adam_conv3_layer_bias_m:@<
)assignvariableop_27_adam_dense_1_kernel_m:	? @5
'assignvariableop_28_adam_dense_1_bias_m:@:
(assignvariableop_29_adam_output_kernel_m:@
4
&assignvariableop_30_adam_output_bias_m:
G
-assignvariableop_31_adam_conv1_layer_kernel_v: 9
+assignvariableop_32_adam_conv1_layer_bias_v: G
-assignvariableop_33_adam_conv2_layer_kernel_v: @9
+assignvariableop_34_adam_conv2_layer_bias_v:@G
-assignvariableop_35_adam_conv3_layer_kernel_v:@@9
+assignvariableop_36_adam_conv3_layer_bias_v:@<
)assignvariableop_37_adam_dense_1_kernel_v:	? @5
'assignvariableop_38_adam_dense_1_bias_v:@:
(assignvariableop_39_adam_output_kernel_v:@
4
&assignvariableop_40_adam_output_bias_v:

identity_42??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:**
dtype0*?
value?B?*B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/1/count/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/2/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-3/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-3/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-4/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-4/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:**
dtype0*g
value^B\*B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::*8
dtypes.
,2*	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOp#assignvariableop_conv1_layer_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOp#assignvariableop_1_conv1_layer_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp%assignvariableop_2_conv2_layer_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp#assignvariableop_3_conv2_layer_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp%assignvariableop_4_conv3_layer_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp#assignvariableop_5_conv3_layer_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp!assignvariableop_6_dense_1_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOpassignvariableop_7_dense_1_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp assignvariableop_8_output_kernelIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOpassignvariableop_9_output_biasIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0	*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOpassignvariableop_10_adam_iterIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOpassignvariableop_11_adam_beta_1Identity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOpassignvariableop_12_adam_beta_2Identity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOpassignvariableop_13_adam_decayIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp&assignvariableop_14_adam_learning_rateIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpassignvariableop_15_totalIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpassignvariableop_16_countIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpassignvariableop_17_total_1Identity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpassignvariableop_18_count_1Identity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpassignvariableop_19_total_2Identity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpassignvariableop_20_count_2Identity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOp-assignvariableop_21_adam_conv1_layer_kernel_mIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOp+assignvariableop_22_adam_conv1_layer_bias_mIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOp-assignvariableop_23_adam_conv2_layer_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp+assignvariableop_24_adam_conv2_layer_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOp-assignvariableop_25_adam_conv3_layer_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOp+assignvariableop_26_adam_conv3_layer_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp)assignvariableop_27_adam_dense_1_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp'assignvariableop_28_adam_dense_1_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOp(assignvariableop_29_adam_output_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp&assignvariableop_30_adam_output_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOp-assignvariableop_31_adam_conv1_layer_kernel_vIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp+assignvariableop_32_adam_conv1_layer_bias_vIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOp-assignvariableop_33_adam_conv2_layer_kernel_vIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOp+assignvariableop_34_adam_conv2_layer_bias_vIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOp-assignvariableop_35_adam_conv3_layer_kernel_vIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOp+assignvariableop_36_adam_conv3_layer_bias_vIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOp)assignvariableop_37_adam_dense_1_kernel_vIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOp'assignvariableop_38_adam_dense_1_bias_vIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOp(assignvariableop_39_adam_output_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp&assignvariableop_40_adam_output_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_409
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_41Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_41?
Identity_42IdentityIdentity_41:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_42"#
identity_42Identity_42:output:0*g
_input_shapesV
T: : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?
a
E__inference_flatt_layer_layer_call_and_return_conditional_losses_4672

inputs
identity_
ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
Consth
ReshapeReshapeinputsConst:output:0*
T0*(
_output_shapes
:?????????? 2	
Reshapee
IdentityIdentityReshape:output:0*
T0*(
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?

?
$__inference_model_layer_call_fn_4581

inputs!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@
	unknown_5:	? @
	unknown_6:@
	unknown_7:@

	unknown_8:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_41892
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????  : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
c
G__inference_max_pooling_1_layer_call_and_return_conditional_losses_4069

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
??
?
__inference__wrapped_model_4063
input_1J
0model_conv1_layer_conv2d_readvariableop_resource: ?
1model_conv1_layer_biasadd_readvariableop_resource: J
0model_conv2_layer_conv2d_readvariableop_resource: @?
1model_conv2_layer_biasadd_readvariableop_resource:@J
0model_conv3_layer_conv2d_readvariableop_resource:@@?
1model_conv3_layer_biasadd_readvariableop_resource:@?
,model_dense_1_matmul_readvariableop_resource:	? @;
-model_dense_1_biasadd_readvariableop_resource:@=
+model_output_matmul_readvariableop_resource:@
:
,model_output_biasadd_readvariableop_resource:

identity??(model/conv1_layer/BiasAdd/ReadVariableOp?'model/conv1_layer/Conv2D/ReadVariableOp?(model/conv2_layer/BiasAdd/ReadVariableOp?'model/conv2_layer/Conv2D/ReadVariableOp?(model/conv3_layer/BiasAdd/ReadVariableOp?'model/conv3_layer/Conv2D/ReadVariableOp?$model/dense_1/BiasAdd/ReadVariableOp?#model/dense_1/MatMul/ReadVariableOp?#model/output/BiasAdd/ReadVariableOp?"model/output/MatMul/ReadVariableOp?
'model/conv1_layer/Conv2D/ReadVariableOpReadVariableOp0model_conv1_layer_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02)
'model/conv1_layer/Conv2D/ReadVariableOp?
model/conv1_layer/Conv2DConv2Dinput_1/model/conv1_layer/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
model/conv1_layer/Conv2D?
(model/conv1_layer/BiasAdd/ReadVariableOpReadVariableOp1model_conv1_layer_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02*
(model/conv1_layer/BiasAdd/ReadVariableOp?
model/conv1_layer/BiasAddBiasAdd!model/conv1_layer/Conv2D:output:00model/conv1_layer/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2
model/conv1_layer/BiasAdd?
model/conv1_layer/ReluRelu"model/conv1_layer/BiasAdd:output:0*
T0*/
_output_shapes
:?????????   2
model/conv1_layer/Relu?
model/max_pooling_1/MaxPoolMaxPool$model/conv1_layer/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
model/max_pooling_1/MaxPool?
'model/conv2_layer/Conv2D/ReadVariableOpReadVariableOp0model_conv2_layer_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02)
'model/conv2_layer/Conv2D/ReadVariableOp?
model/conv2_layer/Conv2DConv2D$model/max_pooling_1/MaxPool:output:0/model/conv2_layer/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
model/conv2_layer/Conv2D?
(model/conv2_layer/BiasAdd/ReadVariableOpReadVariableOp1model_conv2_layer_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model/conv2_layer/BiasAdd/ReadVariableOp?
model/conv2_layer/BiasAddBiasAdd!model/conv2_layer/Conv2D:output:00model/conv2_layer/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
model/conv2_layer/BiasAdd?
model/conv2_layer/ReluRelu"model/conv2_layer/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
model/conv2_layer/Relu?
model/max_pooling_2/MaxPoolMaxPool$model/conv2_layer/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
model/max_pooling_2/MaxPool?
'model/conv3_layer/Conv2D/ReadVariableOpReadVariableOp0model_conv3_layer_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02)
'model/conv3_layer/Conv2D/ReadVariableOp?
model/conv3_layer/Conv2DConv2D$model/max_pooling_2/MaxPool:output:0/model/conv3_layer/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
model/conv3_layer/Conv2D?
(model/conv3_layer/BiasAdd/ReadVariableOpReadVariableOp1model_conv3_layer_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02*
(model/conv3_layer/BiasAdd/ReadVariableOp?
model/conv3_layer/BiasAddBiasAdd!model/conv3_layer/Conv2D:output:00model/conv3_layer/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
model/conv3_layer/BiasAdd?
model/conv3_layer/ReluRelu"model/conv3_layer/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
model/conv3_layer/Relu?
model/flatt_layer/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
model/flatt_layer/Const?
model/flatt_layer/ReshapeReshape$model/conv3_layer/Relu:activations:0 model/flatt_layer/Const:output:0*
T0*(
_output_shapes
:?????????? 2
model/flatt_layer/Reshape?
#model/dense_1/MatMul/ReadVariableOpReadVariableOp,model_dense_1_matmul_readvariableop_resource*
_output_shapes
:	? @*
dtype02%
#model/dense_1/MatMul/ReadVariableOp?
model/dense_1/MatMulMatMul"model/flatt_layer/Reshape:output:0+model/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model/dense_1/MatMul?
$model/dense_1/BiasAdd/ReadVariableOpReadVariableOp-model_dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02&
$model/dense_1/BiasAdd/ReadVariableOp?
model/dense_1/BiasAddBiasAddmodel/dense_1/MatMul:product:0,model/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
model/dense_1/BiasAdd?
model/dense_1/ReluRelumodel/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
model/dense_1/Relu?
"model/output/MatMul/ReadVariableOpReadVariableOp+model_output_matmul_readvariableop_resource*
_output_shapes

:@
*
dtype02$
"model/output/MatMul/ReadVariableOp?
model/output/MatMulMatMul model/dense_1/Relu:activations:0*model/output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
model/output/MatMul?
#model/output/BiasAdd/ReadVariableOpReadVariableOp,model_output_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02%
#model/output/BiasAdd/ReadVariableOp?
model/output/BiasAddBiasAddmodel/output/MatMul:product:0+model/output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
model/output/BiasAdd?
IdentityIdentitymodel/output/BiasAdd:output:0)^model/conv1_layer/BiasAdd/ReadVariableOp(^model/conv1_layer/Conv2D/ReadVariableOp)^model/conv2_layer/BiasAdd/ReadVariableOp(^model/conv2_layer/Conv2D/ReadVariableOp)^model/conv3_layer/BiasAdd/ReadVariableOp(^model/conv3_layer/Conv2D/ReadVariableOp%^model/dense_1/BiasAdd/ReadVariableOp$^model/dense_1/MatMul/ReadVariableOp$^model/output/BiasAdd/ReadVariableOp#^model/output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????  : : : : : : : : : : 2T
(model/conv1_layer/BiasAdd/ReadVariableOp(model/conv1_layer/BiasAdd/ReadVariableOp2R
'model/conv1_layer/Conv2D/ReadVariableOp'model/conv1_layer/Conv2D/ReadVariableOp2T
(model/conv2_layer/BiasAdd/ReadVariableOp(model/conv2_layer/BiasAdd/ReadVariableOp2R
'model/conv2_layer/Conv2D/ReadVariableOp'model/conv2_layer/Conv2D/ReadVariableOp2T
(model/conv3_layer/BiasAdd/ReadVariableOp(model/conv3_layer/BiasAdd/ReadVariableOp2R
'model/conv3_layer/Conv2D/ReadVariableOp'model/conv3_layer/Conv2D/ReadVariableOp2L
$model/dense_1/BiasAdd/ReadVariableOp$model/dense_1/BiasAdd/ReadVariableOp2J
#model/dense_1/MatMul/ReadVariableOp#model/dense_1/MatMul/ReadVariableOp2J
#model/output/BiasAdd/ReadVariableOp#model/output/BiasAdd/ReadVariableOp2H
"model/output/MatMul/ReadVariableOp"model/output/MatMul/ReadVariableOp:X T
/
_output_shapes
:?????????  
!
_user_specified_name	input_1
?
H
,__inference_max_pooling_2_layer_call_fn_4087

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *J
_output_shapes8
6:4????????????????????????????????????* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_max_pooling_2_layer_call_and_return_conditional_losses_40812
PartitionedCall?
IdentityIdentityPartitionedCall:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
$__inference_model_layer_call_fn_4606

inputs!
unknown: 
	unknown_0: #
	unknown_1: @
	unknown_2:@#
	unknown_3:@@
	unknown_4:@
	unknown_5:	? @
	unknown_6:@
	unknown_7:@

	unknown_8:

identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*,
_read_only_resource_inputs

	
*-
config_proto

CPU

GPU 2J 8? *H
fCRA
?__inference_model_layer_call_and_return_conditional_losses_43272
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????  : : : : : : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
E__inference_conv1_layer_layer_call_and_return_conditional_losses_4617

inputs8
conv2d_readvariableop_resource: -
biasadd_readvariableop_resource: 
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????   2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????   2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????  : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
E__inference_conv3_layer_layer_call_and_return_conditional_losses_4141

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?
?
E__inference_conv3_layer_layer_call_and_return_conditional_losses_4657

inputs8
conv2d_readvariableop_resource:@@-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????@: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs
?&
?
?__inference_model_layer_call_and_return_conditional_losses_4327

inputs*
conv1_layer_4298: 
conv1_layer_4300: *
conv2_layer_4304: @
conv2_layer_4306:@*
conv3_layer_4310:@@
conv3_layer_4312:@
dense_1_4316:	? @
dense_1_4318:@
output_4321:@

output_4323:

identity??#conv1_layer/StatefulPartitionedCall?#conv2_layer/StatefulPartitionedCall?#conv3_layer/StatefulPartitionedCall?dense_1/StatefulPartitionedCall?output/StatefulPartitionedCall?
#conv1_layer/StatefulPartitionedCallStatefulPartitionedCallinputsconv1_layer_4298conv1_layer_4300*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????   *$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv1_layer_layer_call_and_return_conditional_losses_41052%
#conv1_layer/StatefulPartitionedCall?
max_pooling_1/PartitionedCallPartitionedCall,conv1_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_max_pooling_1_layer_call_and_return_conditional_losses_40692
max_pooling_1/PartitionedCall?
#conv2_layer/StatefulPartitionedCallStatefulPartitionedCall&max_pooling_1/PartitionedCall:output:0conv2_layer_4304conv2_layer_4306*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv2_layer_layer_call_and_return_conditional_losses_41232%
#conv2_layer/StatefulPartitionedCall?
max_pooling_2/PartitionedCallPartitionedCall,conv2_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@* 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *P
fKRI
G__inference_max_pooling_2_layer_call_and_return_conditional_losses_40812
max_pooling_2/PartitionedCall?
#conv3_layer/StatefulPartitionedCallStatefulPartitionedCall&max_pooling_2/PartitionedCall:output:0conv3_layer_4310conv3_layer_4312*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_conv3_layer_layer_call_and_return_conditional_losses_41412%
#conv3_layer/StatefulPartitionedCall?
flatt_layer/PartitionedCallPartitionedCall,conv3_layer/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatt_layer_layer_call_and_return_conditional_losses_41532
flatt_layer/PartitionedCall?
dense_1/StatefulPartitionedCallStatefulPartitionedCall$flatt_layer/PartitionedCall:output:0dense_1_4316dense_1_4318*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_41662!
dense_1/StatefulPartitionedCall?
output/StatefulPartitionedCallStatefulPartitionedCall(dense_1/StatefulPartitionedCall:output:0output_4321output_4323*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????
*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *I
fDRB
@__inference_output_layer_call_and_return_conditional_losses_41822 
output/StatefulPartitionedCall?
IdentityIdentity'output/StatefulPartitionedCall:output:0$^conv1_layer/StatefulPartitionedCall$^conv2_layer/StatefulPartitionedCall$^conv3_layer/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall^output/StatefulPartitionedCall*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????  : : : : : : : : : : 2J
#conv1_layer/StatefulPartitionedCall#conv1_layer/StatefulPartitionedCall2J
#conv2_layer/StatefulPartitionedCall#conv2_layer/StatefulPartitionedCall2J
#conv3_layer/StatefulPartitionedCall#conv3_layer/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall2@
output/StatefulPartitionedCalloutput/StatefulPartitionedCall:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
?
&__inference_dense_1_layer_call_fn_4697

inputs
unknown:	? @
	unknown_0:@
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????@*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8? *J
fERC
A__inference_dense_1_layer_call_and_return_conditional_losses_41662
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*+
_input_shapes
:?????????? : : 22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:?????????? 
 
_user_specified_nameinputs
?
c
G__inference_max_pooling_2_layer_call_and_return_conditional_losses_4081

inputs
identity?
MaxPoolMaxPoolinputs*J
_output_shapes8
6:4????????????????????????????????????*
ksize
*
paddingVALID*
strides
2	
MaxPool?
IdentityIdentityMaxPool:output:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*I
_input_shapes8
6:4????????????????????????????????????:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
E__inference_conv2_layer_layer_call_and_return_conditional_losses_4637

inputs8
conv2d_readvariableop_resource: @-
biasadd_readvariableop_resource:@
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2	
BiasAdd`
ReluReluBiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*/
_output_shapes
:?????????@2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:????????? : : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:????????? 
 
_user_specified_nameinputs
?8
?
?__inference_model_layer_call_and_return_conditional_losses_4556

inputsD
*conv1_layer_conv2d_readvariableop_resource: 9
+conv1_layer_biasadd_readvariableop_resource: D
*conv2_layer_conv2d_readvariableop_resource: @9
+conv2_layer_biasadd_readvariableop_resource:@D
*conv3_layer_conv2d_readvariableop_resource:@@9
+conv3_layer_biasadd_readvariableop_resource:@9
&dense_1_matmul_readvariableop_resource:	? @5
'dense_1_biasadd_readvariableop_resource:@7
%output_matmul_readvariableop_resource:@
4
&output_biasadd_readvariableop_resource:

identity??"conv1_layer/BiasAdd/ReadVariableOp?!conv1_layer/Conv2D/ReadVariableOp?"conv2_layer/BiasAdd/ReadVariableOp?!conv2_layer/Conv2D/ReadVariableOp?"conv3_layer/BiasAdd/ReadVariableOp?!conv3_layer/Conv2D/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?output/BiasAdd/ReadVariableOp?output/MatMul/ReadVariableOp?
!conv1_layer/Conv2D/ReadVariableOpReadVariableOp*conv1_layer_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02#
!conv1_layer/Conv2D/ReadVariableOp?
conv1_layer/Conv2DConv2Dinputs)conv1_layer/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   *
paddingSAME*
strides
2
conv1_layer/Conv2D?
"conv1_layer/BiasAdd/ReadVariableOpReadVariableOp+conv1_layer_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02$
"conv1_layer/BiasAdd/ReadVariableOp?
conv1_layer/BiasAddBiasAddconv1_layer/Conv2D:output:0*conv1_layer/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????   2
conv1_layer/BiasAdd?
conv1_layer/ReluReluconv1_layer/BiasAdd:output:0*
T0*/
_output_shapes
:?????????   2
conv1_layer/Relu?
max_pooling_1/MaxPoolMaxPoolconv1_layer/Relu:activations:0*/
_output_shapes
:????????? *
ksize
*
paddingVALID*
strides
2
max_pooling_1/MaxPool?
!conv2_layer/Conv2D/ReadVariableOpReadVariableOp*conv2_layer_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02#
!conv2_layer/Conv2D/ReadVariableOp?
conv2_layer/Conv2DConv2Dmax_pooling_1/MaxPool:output:0)conv2_layer/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv2_layer/Conv2D?
"conv2_layer/BiasAdd/ReadVariableOpReadVariableOp+conv2_layer_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02$
"conv2_layer/BiasAdd/ReadVariableOp?
conv2_layer/BiasAddBiasAddconv2_layer/Conv2D:output:0*conv2_layer/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2_layer/BiasAdd?
conv2_layer/ReluReluconv2_layer/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2_layer/Relu?
max_pooling_2/MaxPoolMaxPoolconv2_layer/Relu:activations:0*/
_output_shapes
:?????????@*
ksize
*
paddingVALID*
strides
2
max_pooling_2/MaxPool?
!conv3_layer/Conv2D/ReadVariableOpReadVariableOp*conv3_layer_conv2d_readvariableop_resource*&
_output_shapes
:@@*
dtype02#
!conv3_layer/Conv2D/ReadVariableOp?
conv3_layer/Conv2DConv2Dmax_pooling_2/MaxPool:output:0)conv3_layer/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2
conv3_layer/Conv2D?
"conv3_layer/BiasAdd/ReadVariableOpReadVariableOp+conv3_layer_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02$
"conv3_layer/BiasAdd/ReadVariableOp?
conv3_layer/BiasAddBiasAddconv3_layer/Conv2D:output:0*conv3_layer/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv3_layer/BiasAdd?
conv3_layer/ReluReluconv3_layer/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv3_layer/Reluw
flatt_layer/ConstConst*
_output_shapes
:*
dtype0*
valueB"????   2
flatt_layer/Const?
flatt_layer/ReshapeReshapeconv3_layer/Relu:activations:0flatt_layer/Const:output:0*
T0*(
_output_shapes
:?????????? 2
flatt_layer/Reshape?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	? @*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMulflatt_layer/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????@2
dense_1/BiasAddp
dense_1/ReluReludense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????@2
dense_1/Relu?
output/MatMul/ReadVariableOpReadVariableOp%output_matmul_readvariableop_resource*
_output_shapes

:@
*
dtype02
output/MatMul/ReadVariableOp?
output/MatMulMatMuldense_1/Relu:activations:0$output/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
output/MatMul?
output/BiasAdd/ReadVariableOpReadVariableOp&output_biasadd_readvariableop_resource*
_output_shapes
:
*
dtype02
output/BiasAdd/ReadVariableOp?
output/BiasAddBiasAddoutput/MatMul:product:0%output/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????
2
output/BiasAdd?
IdentityIdentityoutput/BiasAdd:output:0#^conv1_layer/BiasAdd/ReadVariableOp"^conv1_layer/Conv2D/ReadVariableOp#^conv2_layer/BiasAdd/ReadVariableOp"^conv2_layer/Conv2D/ReadVariableOp#^conv3_layer/BiasAdd/ReadVariableOp"^conv3_layer/Conv2D/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp^output/BiasAdd/ReadVariableOp^output/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????
2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*B
_input_shapes1
/:?????????  : : : : : : : : : : 2H
"conv1_layer/BiasAdd/ReadVariableOp"conv1_layer/BiasAdd/ReadVariableOp2F
!conv1_layer/Conv2D/ReadVariableOp!conv1_layer/Conv2D/ReadVariableOp2H
"conv2_layer/BiasAdd/ReadVariableOp"conv2_layer/BiasAdd/ReadVariableOp2F
!conv2_layer/Conv2D/ReadVariableOp!conv2_layer/Conv2D/ReadVariableOp2H
"conv3_layer/BiasAdd/ReadVariableOp"conv3_layer/BiasAdd/ReadVariableOp2F
!conv3_layer/Conv2D/ReadVariableOp!conv3_layer/Conv2D/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp2>
output/BiasAdd/ReadVariableOpoutput/BiasAdd/ReadVariableOp2<
output/MatMul/ReadVariableOpoutput/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????  
 
_user_specified_nameinputs
?
F
*__inference_flatt_layer_layer_call_fn_4677

inputs
identity?
PartitionedCallPartitionedCallinputs*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:?????????? * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8? *N
fIRG
E__inference_flatt_layer_layer_call_and_return_conditional_losses_41532
PartitionedCallm
IdentityIdentityPartitionedCall:output:0*
T0*(
_output_shapes
:?????????? 2

Identity"
identityIdentity:output:0*(
_construction_contextkEagerRuntime*.
_input_shapes
:?????????@:W S
/
_output_shapes
:?????????@
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
input_18
serving_default_input_1:0?????????  :
output0
StatefulPartitionedCall:0?????????
tensorflow/serving/predict:ϯ
?Y
layer-0
layer_with_weights-0
layer-1
layer-2
layer_with_weights-1
layer-3
layer-4
layer_with_weights-2
layer-5
layer-6
layer_with_weights-3
layer-7
	layer_with_weights-4
	layer-8

	optimizer
trainable_variables
regularization_losses
	variables
	keras_api

signatures
+?&call_and_return_all_conditional_losses
?_default_save_signature
?__call__"?U
_tf_keras_network?U{"name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv1_layer", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1_layer", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling_1", "inbound_nodes": [[["conv1_layer", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2_layer", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2_layer", "inbound_nodes": [[["max_pooling_1", 0, 0, {}]]]}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling_2", "inbound_nodes": [[["conv2_layer", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv3_layer", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3_layer", "inbound_nodes": [[["max_pooling_2", 0, 0, {}]]]}, {"class_name": "Flatten", "config": {"name": "flatt_layer", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatt_layer", "inbound_nodes": [[["conv3_layer", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["flatt_layer", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 10, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["dense_1", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["output", 0, 0]]}, "shared_object_id": 19, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 3]}, "is_graph_network": true, "save_spec": {"class_name": "TypeSpec", "type_spec": "tf.TensorSpec", "serialized": [{"class_name": "TensorShape", "items": [null, 32, 32, 3]}, "float32", "input_1"]}, "keras_version": "2.5.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": [], "shared_object_id": 0}, {"class_name": "Conv2D", "config": {"name": "conv1_layer", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv1_layer", "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 3}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling_1", "inbound_nodes": [[["conv1_layer", 0, 0, {}]]], "shared_object_id": 4}, {"class_name": "Conv2D", "config": {"name": "conv2_layer", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2_layer", "inbound_nodes": [[["max_pooling_1", 0, 0, {}]]], "shared_object_id": 7}, {"class_name": "MaxPooling2D", "config": {"name": "max_pooling_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "name": "max_pooling_2", "inbound_nodes": [[["conv2_layer", 0, 0, {}]]], "shared_object_id": 8}, {"class_name": "Conv2D", "config": {"name": "conv3_layer", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv3_layer", "inbound_nodes": [[["max_pooling_2", 0, 0, {}]]], "shared_object_id": 11}, {"class_name": "Flatten", "config": {"name": "flatt_layer", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "name": "flatt_layer", "inbound_nodes": [[["conv3_layer", 0, 0, {}]]], "shared_object_id": 12}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["flatt_layer", 0, 0, {}]]], "shared_object_id": 15}, {"class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 10, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 16}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "output", "inbound_nodes": [[["dense_1", 0, 0, {}]]], "shared_object_id": 18}], "input_layers": [["input_1", 0, 0]], "output_layers": [["output", 0, 0]]}}, "training_config": {"loss": {"class_name": "SparseCategoricalCrossentropy", "config": {"reduction": "auto", "name": "sparse_categorical_crossentropy", "from_logits": true}, "shared_object_id": 21}, "metrics": [[{"class_name": "SparseCategoricalCrossentropy", "config": {"name": "CrossEntropy", "dtype": "float32", "from_logits": false, "axis": -1}, "shared_object_id": 22}, {"class_name": "MeanMetricWrapper", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}, "shared_object_id": 23}]], "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 0.0010000000474974513, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 32, 32, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "conv1_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv1_layer", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 1}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 2}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["input_1", 0, 0, {}]]], "shared_object_id": 3, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 3}}, "shared_object_id": 24}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 32, 32, 3]}}
?
trainable_variables
	variables
regularization_losses
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "max_pooling_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling_1", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "inbound_nodes": [[["conv1_layer", 0, 0, {}]]], "shared_object_id": 4, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 25}}
?

kernel
bias
trainable_variables
	variables
regularization_losses
	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "conv2_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv2_layer", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 5}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 6}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["max_pooling_1", 0, 0, {}]]], "shared_object_id": 7, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}, "shared_object_id": 26}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 16, 16, 32]}}
?
 trainable_variables
!	variables
"regularization_losses
#	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "max_pooling_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "MaxPooling2D", "config": {"name": "max_pooling_2", "trainable": true, "dtype": "float32", "pool_size": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "strides": {"class_name": "__tuple__", "items": [2, 2]}, "data_format": "channels_last"}, "inbound_nodes": [[["conv2_layer", 0, 0, {}]]], "shared_object_id": 8, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}, "shared_object_id": 27}}
?

$kernel
%bias
&trainable_variables
'	variables
(regularization_losses
)	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?	
_tf_keras_layer?	{"name": "conv3_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Conv2D", "config": {"name": "conv3_layer", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [4, 4]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 9}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 10}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["max_pooling_2", 0, 0, {}]]], "shared_object_id": 11, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 64}}, "shared_object_id": 28}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 8, 8, 64]}}
?
*trainable_variables
+	variables
,regularization_losses
-	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "flatt_layer", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Flatten", "config": {"name": "flatt_layer", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "inbound_nodes": [[["conv3_layer", 0, 0, {}]]], "shared_object_id": 12, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}, "shared_object_id": 29}}
?	

.kernel
/bias
0trainable_variables
1	variables
2regularization_losses
3	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 64, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 13}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 14}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["flatt_layer", 0, 0, {}]]], "shared_object_id": 15, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 4096}}, "shared_object_id": 30}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 4096]}}
?

4kernel
5bias
6trainable_variables
7	variables
8regularization_losses
9	keras_api
+?&call_and_return_all_conditional_losses
?__call__"?
_tf_keras_layer?{"name": "output", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "class_name": "Dense", "config": {"name": "output", "trainable": true, "dtype": "float32", "units": 10, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}, "shared_object_id": 16}, "bias_initializer": {"class_name": "Zeros", "config": {}, "shared_object_id": 17}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "inbound_nodes": [[["dense_1", 0, 0, {}]]], "shared_object_id": 18, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 64}}, "shared_object_id": 31}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64]}}
?
:iter

;beta_1

<beta_2
	=decay
>learning_ratem}m~mm?$m?%m?.m?/m?4m?5m?v?v?v?v?$v?%v?.v?/v?4v?5v?"
	optimizer
f
0
1
2
3
$4
%5
.6
/7
48
59"
trackable_list_wrapper
 "
trackable_list_wrapper
f
0
1
2
3
$4
%5
.6
/7
48
59"
trackable_list_wrapper
?
?layer_regularization_losses
trainable_variables
@non_trainable_variables
Alayer_metrics

Blayers
regularization_losses
Cmetrics
	variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
,:* 2conv1_layer/kernel
: 2conv1_layer/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Dlayer_regularization_losses
trainable_variables
Enon_trainable_variables
Flayer_metrics

Glayers
	variables
Hmetrics
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Ilayer_regularization_losses
trainable_variables
Jnon_trainable_variables
Klayer_metrics

Llayers
	variables
Mmetrics
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
,:* @2conv2_layer/kernel
:@2conv2_layer/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Nlayer_regularization_losses
trainable_variables
Onon_trainable_variables
Player_metrics

Qlayers
	variables
Rmetrics
regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
Slayer_regularization_losses
 trainable_variables
Tnon_trainable_variables
Ulayer_metrics

Vlayers
!	variables
Wmetrics
"regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
,:*@@2conv3_layer/kernel
:@2conv3_layer/bias
.
$0
%1"
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
Xlayer_regularization_losses
&trainable_variables
Ynon_trainable_variables
Zlayer_metrics

[layers
'	variables
\metrics
(regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
]layer_regularization_losses
*trainable_variables
^non_trainable_variables
_layer_metrics

`layers
+	variables
ametrics
,regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
!:	? @2dense_1/kernel
:@2dense_1/bias
.
.0
/1"
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
?
blayer_regularization_losses
0trainable_variables
cnon_trainable_variables
dlayer_metrics

elayers
1	variables
fmetrics
2regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:@
2output/kernel
:
2output/bias
.
40
51"
trackable_list_wrapper
.
40
51"
trackable_list_wrapper
 "
trackable_list_wrapper
?
glayer_regularization_losses
6trainable_variables
hnon_trainable_variables
ilayer_metrics

jlayers
7	variables
kmetrics
8regularization_losses
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
_
0
1
2
3
4
5
6
7
	8"
trackable_list_wrapper
5
l0
m1
n2"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
	ototal
	pcount
q	variables
r	keras_api"?
_tf_keras_metric?{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}, "shared_object_id": 32}
?
	stotal
	tcount
u
_fn_kwargs
v	variables
w	keras_api"?
_tf_keras_metric?{"class_name": "SparseCategoricalCrossentropy", "name": "CrossEntropy", "dtype": "float32", "config": {"name": "CrossEntropy", "dtype": "float32", "from_logits": false, "axis": -1}, "shared_object_id": 22}
?
	xtotal
	ycount
z
_fn_kwargs
{	variables
|	keras_api"?
_tf_keras_metric?{"class_name": "MeanMetricWrapper", "name": "accuracy", "dtype": "float32", "config": {"name": "accuracy", "dtype": "float32", "fn": "sparse_categorical_accuracy"}, "shared_object_id": 23}
:  (2total
:  (2count
.
o0
p1"
trackable_list_wrapper
-
q	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
s0
t1"
trackable_list_wrapper
-
v	variables"
_generic_user_object
:  (2total
:  (2count
 "
trackable_dict_wrapper
.
x0
y1"
trackable_list_wrapper
-
{	variables"
_generic_user_object
1:/ 2Adam/conv1_layer/kernel/m
#:! 2Adam/conv1_layer/bias/m
1:/ @2Adam/conv2_layer/kernel/m
#:!@2Adam/conv2_layer/bias/m
1:/@@2Adam/conv3_layer/kernel/m
#:!@2Adam/conv3_layer/bias/m
&:$	? @2Adam/dense_1/kernel/m
:@2Adam/dense_1/bias/m
$:"@
2Adam/output/kernel/m
:
2Adam/output/bias/m
1:/ 2Adam/conv1_layer/kernel/v
#:! 2Adam/conv1_layer/bias/v
1:/ @2Adam/conv2_layer/kernel/v
#:!@2Adam/conv2_layer/bias/v
1:/@@2Adam/conv3_layer/kernel/v
#:!@2Adam/conv3_layer/bias/v
&:$	? @2Adam/dense_1/kernel/v
:@2Adam/dense_1/bias/v
$:"@
2Adam/output/kernel/v
:
2Adam/output/bias/v
?2?
?__inference_model_layer_call_and_return_conditional_losses_4514
?__inference_model_layer_call_and_return_conditional_losses_4556
?__inference_model_layer_call_and_return_conditional_losses_4407
?__inference_model_layer_call_and_return_conditional_losses_4439?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
__inference__wrapped_model_4063?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *.?+
)?&
input_1?????????  
?2?
$__inference_model_layer_call_fn_4212
$__inference_model_layer_call_fn_4581
$__inference_model_layer_call_fn_4606
$__inference_model_layer_call_fn_4375?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
E__inference_conv1_layer_layer_call_and_return_conditional_losses_4617?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv1_layer_layer_call_fn_4626?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_max_pooling_1_layer_call_and_return_conditional_losses_4069?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
,__inference_max_pooling_1_layer_call_fn_4075?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
E__inference_conv2_layer_layer_call_and_return_conditional_losses_4637?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv2_layer_layer_call_fn_4646?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
G__inference_max_pooling_2_layer_call_and_return_conditional_losses_4081?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
,__inference_max_pooling_2_layer_call_fn_4087?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
E__inference_conv3_layer_layer_call_and_return_conditional_losses_4657?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_conv3_layer_layer_call_fn_4666?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
E__inference_flatt_layer_layer_call_and_return_conditional_losses_4672?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_flatt_layer_layer_call_fn_4677?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_dense_1_layer_call_and_return_conditional_losses_4688?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
&__inference_dense_1_layer_call_fn_4697?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
@__inference_output_layer_call_and_return_conditional_losses_4707?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
%__inference_output_layer_call_fn_4716?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?B?
"__inference_signature_wrapper_4472input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
__inference__wrapped_model_4063w
$%./458?5
.?+
)?&
input_1?????????  
? "/?,
*
output ?
output?????????
?
E__inference_conv1_layer_layer_call_and_return_conditional_losses_4617l7?4
-?*
(?%
inputs?????????  
? "-?*
#? 
0?????????   
? ?
*__inference_conv1_layer_layer_call_fn_4626_7?4
-?*
(?%
inputs?????????  
? " ??????????   ?
E__inference_conv2_layer_layer_call_and_return_conditional_losses_4637l7?4
-?*
(?%
inputs????????? 
? "-?*
#? 
0?????????@
? ?
*__inference_conv2_layer_layer_call_fn_4646_7?4
-?*
(?%
inputs????????? 
? " ??????????@?
E__inference_conv3_layer_layer_call_and_return_conditional_losses_4657l$%7?4
-?*
(?%
inputs?????????@
? "-?*
#? 
0?????????@
? ?
*__inference_conv3_layer_layer_call_fn_4666_$%7?4
-?*
(?%
inputs?????????@
? " ??????????@?
A__inference_dense_1_layer_call_and_return_conditional_losses_4688]./0?-
&?#
!?
inputs?????????? 
? "%?"
?
0?????????@
? z
&__inference_dense_1_layer_call_fn_4697P./0?-
&?#
!?
inputs?????????? 
? "??????????@?
E__inference_flatt_layer_layer_call_and_return_conditional_losses_4672a7?4
-?*
(?%
inputs?????????@
? "&?#
?
0?????????? 
? ?
*__inference_flatt_layer_layer_call_fn_4677T7?4
-?*
(?%
inputs?????????@
? "??????????? ?
G__inference_max_pooling_1_layer_call_and_return_conditional_losses_4069?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
,__inference_max_pooling_1_layer_call_fn_4075?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
G__inference_max_pooling_2_layer_call_and_return_conditional_losses_4081?R?O
H?E
C?@
inputs4????????????????????????????????????
? "H?E
>?;
04????????????????????????????????????
? ?
,__inference_max_pooling_2_layer_call_fn_4087?R?O
H?E
C?@
inputs4????????????????????????????????????
? ";?84?????????????????????????????????????
?__inference_model_layer_call_and_return_conditional_losses_4407u
$%./45@?=
6?3
)?&
input_1?????????  
p 

 
? "%?"
?
0?????????

? ?
?__inference_model_layer_call_and_return_conditional_losses_4439u
$%./45@?=
6?3
)?&
input_1?????????  
p

 
? "%?"
?
0?????????

? ?
?__inference_model_layer_call_and_return_conditional_losses_4514t
$%./45??<
5?2
(?%
inputs?????????  
p 

 
? "%?"
?
0?????????

? ?
?__inference_model_layer_call_and_return_conditional_losses_4556t
$%./45??<
5?2
(?%
inputs?????????  
p

 
? "%?"
?
0?????????

? ?
$__inference_model_layer_call_fn_4212h
$%./45@?=
6?3
)?&
input_1?????????  
p 

 
? "??????????
?
$__inference_model_layer_call_fn_4375h
$%./45@?=
6?3
)?&
input_1?????????  
p

 
? "??????????
?
$__inference_model_layer_call_fn_4581g
$%./45??<
5?2
(?%
inputs?????????  
p 

 
? "??????????
?
$__inference_model_layer_call_fn_4606g
$%./45??<
5?2
(?%
inputs?????????  
p

 
? "??????????
?
@__inference_output_layer_call_and_return_conditional_losses_4707\45/?,
%?"
 ?
inputs?????????@
? "%?"
?
0?????????

? x
%__inference_output_layer_call_fn_4716O45/?,
%?"
 ?
inputs?????????@
? "??????????
?
"__inference_signature_wrapper_4472?
$%./45C?@
? 
9?6
4
input_1)?&
input_1?????????  "/?,
*
output ?
output?????????
