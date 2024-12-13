#pragma once



#include "bits/begin_end.hpp"
#include "bits/traits.hpp"
#include "bits/copy.hpp"

#include "bits/range.hpp"
#include "bits/transform_range.hpp"
#include "bits/zip_range.hpp"
#include "bits/constant_range.hpp"

#include "bits/md_range.hpp"
#include "bits/md_transform_range.hpp"
#include "bits/md_zip_range.hpp"
#include "bits/md_constant_range.hpp"


//#include "bits/arithmetic_ops.hpp"
//#include "bits/parallel_force_evaluate.hpp"
//#include "bits/md_transform.hpp"
//#include "bits/md_smart_transform.hpp"
//#include "bits/md_traits.hpp"
//#include "bits/md_numeric_array.hpp"
//#include "bits/zip.hpp"
//#include "bits/transform.hpp"
//#include "bits/smart_transform.hpp"

//#include "bits/numeric_array.hpp"


#ifdef __NVIDIA_COMPILER__
#include "bits/device_host_copy.hpp"
#endif
