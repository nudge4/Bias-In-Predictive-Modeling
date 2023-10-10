use "D:\Yifeng -- Project Work\ys8mz_sandbox\bias_analyses\train_df.dta", clear
keep vccsid strm college course section grade
merge m:1 vccsid strm using "D:\Yifeng -- Project Work\ys8mz_sandbox\bias_analyses\zip_5digits.dta", keep(1 3) nogen
merge m:1 vccsid using "D:\Yifeng -- Project Work\ys8mz_sandbox\bias_analyses\race.dta", keep(1 3) keepusing(male white afam hisp asian other) nogen
merge m:1 vccsid using "D:\Yifeng -- Project Work\ys8mz_sandbox\bias_analyses\firstgen.dta", keep(1 3) nogen
merge m:1 vccsid strm using "D:\Yifeng -- Project Work\ys8mz_sandbox\bias_analyses\pell.dta", keep(1 3) nogen
merge m:1 vccsid strm using "D:\Yifeng -- Project Work\ys8mz_sandbox\bias_analyses\zip.dta", keep(1 3) keepusing(distance) nogen
order vccsid strm college course section grade zip_code is_va_zip distance male white afam hisp asian other firstgen_0 firstgen_1 pell_ever_0 pell_ever_1 pell_target_0 pell_target_1
save "D:\Yifeng -- Project Work\ys8mz_sandbox\bias_analyses\training_set_demographics.dta", replace

use "D:\Yifeng -- Project Work\ys8mz_sandbox\bias_analyses\test_df.dta", clear
keep vccsid strm college course section grade
merge m:1 vccsid strm using "D:\Yifeng -- Project Work\ys8mz_sandbox\bias_analyses\zip_5digits.dta", keep(1 3) nogen
merge m:1 vccsid using "D:\Yifeng -- Project Work\ys8mz_sandbox\bias_analyses\race.dta", keep(1 3) keepusing(male white afam hisp asian other) nogen
merge m:1 vccsid using "D:\Yifeng -- Project Work\ys8mz_sandbox\bias_analyses\firstgen.dta", keep(1 3) nogen
merge m:1 vccsid strm using "D:\Yifeng -- Project Work\ys8mz_sandbox\bias_analyses\pell.dta", keep(1 3) nogen
merge m:1 vccsid strm using "D:\Yifeng -- Project Work\ys8mz_sandbox\bias_analyses\zip.dta", keep(1 3) keepusing(distance) nogen
order vccsid strm college course section grade zip_code is_va_zip distance male white afam hisp asian other firstgen_0 firstgen_1 pell_ever_0 pell_ever_1 pell_target_0 pell_target_1
save "D:\Yifeng -- Project Work\ys8mz_sandbox\bias_analyses\test_set_demographics.dta", replace
