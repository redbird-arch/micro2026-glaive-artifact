The format of workload input is as follows (NOTE that all communication sizes are in bytes and compute times are in terms of cycles):

* **first line**: (DATA/HYBRID_TRANSFORMER/HYBRID_DLRM/DISTRIBUTED_INFERENCE) 
	* The training loop parallelization type. Each new training loop type sould be implemented inside thw Workload.cc file.
	DATA is the pure data-parallel approach. HYBRID_TRANSFORMER is hybrid-parallel tuned for Transformer DNN network. 	  HYBRID_DLRM is hybrid-parallel tuned for DLRM DNN network.

* **second line**: (int)
	* shows the number of DNN layers

* **subsequent lines**: Each subsequent line describes a layer. The format of layer description  is as follows:
	* {(string: **layer name**) (int: **reserved variable**)
	(int: **forward pass compute time**) (ALLREDUCE/ALLGATHER/ALLTOALL/ALLTOALLV: **forward pass communication type**) (int: **forward pass communication size**)
	(int: **input grad compute time**) (ALLREDUCE/ALLGATHER/ALLTOALL/ALLTOALLV: **input grad communication type**) (int: **input grad communication size**)
	(int: **weight grad compute time**) (ALLREDUCE/ALLGATHER/ALLTOALL/ALLTOALLV: **weight grad communication type**) (int: **weight grad communication size**) 
	(**delay per entire weight/input/output update after the collective is finished**)
	[(optional int: **forward memory read bytes**)
	(optional int: **forward memory write bytes**)])} 

*NOTE: all parameters inside the bracket are defined in a single line for each layer of the DNN network.* 
*NOTE: `ALLTOALLV` is currently intended to be resolved through the external collective timing interface rather than Astra's internal analytical all-to-all path.*
*NOTE: the optional forward-memory fields are consumed by `DISTRIBUTED_INFERENCE` to model an explicit pure-memory stage between forward compute and forward communication.*
	 
