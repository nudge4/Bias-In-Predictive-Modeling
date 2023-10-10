/* This script generates a special type of demographic predictors: The first-generation status: first-geneneration, non-first-generation, unknown. */

global data "D:/Yifeng -- Project Work/ys8mz_sandbox"

***************************************************
** Create the parents' highest education predictors
***************************************************
local files: dir "${data}/Build/Student" files "*.dta.zip"
local i = 0
foreach file in `files' {
	if "`file'" != "student_all_records.dta.zip" {
		if `i' == 0 {
			zipuse "${data}/Build/Student/`file'", clear
			keep vccsid fhe mhe
			gen source = `i'
		}
		else {
			preserve
				zipuse "${data}/Build/Student/`file'", clear
				keep vccsid fhe mhe
				gen source = `i'
				tempfile stu_temp_`i'
				save "`stu_temp_`i''", replace
			restore
			append using "`stu_temp_`i''", force
		}
		di "`file'"
		local i = `++i'
	}
}
gsort vccsid -source // relying on Student files during the more recent terms to determine the demographic information if there're discrepancies
replace mhe = 0 if mhe == . // Mother's highest education
replace fhe = 0 if fhe == . // Father's highest education
gen flag_1 = 0
replace flag_1 = 1 if fhe > 0
gen flag_2 = 0
replace flag_2 = 1 if mhe > 0
egen max_source_1 = max(source), by(vccsid)
egen max_source_2 = max(source), by(vccsid)
gen new_fhe_tmp = fhe if source == max_source_1
egen new_fhe = max(new_fhe_tmp), by(vccsid)
gen new_mhe_tmp = mhe if source == max_source_2
egen new_mhe = max(new_mhe_tmp), by(vccsid)
keep vccsid new_fhe new_mhe
duplicates drop
gen max_phe = new_fhe
replace max_phe = new_mhe if new_mhe > new_fhe
gen min_phe = new_fhe
replace min_phe = new_mhe if new_mhe < new_fhe
replace max_phe = 0 if min_phe == 0
keep vccsid max_phe
gen has_phe = 1 // indicator for whether the parents' highest education data are available or not
replace has_phe = 0 if max_phe == 0
rename max_phe phe
order vccsid has_phe
sort vccsid
gen firstgen = (phe <= 3) & (has_phe == 1)
replace firstgen = . if has_phe == 0
keep vccsid firstgen
gen firstgen_0 = 0
gen firstgen_1 = 0
replace firstgen_0 = 1 if firstgen == 0
replace firstgen_1 = 1 if firstgen == 1
drop firstgen

merge 1:1 vccsid using "C:\Users\ys8mz\Box Sync\Predictive Models of College Completion (VCCS)\intermediate_files\new\full_data_truncated.dta", keep(2 3) keepusing(last_term) nogen
save "C:\Users\ys8mz\Box Sync\Predictive Models of College Completion (VCCS)\intermediate_files\new\firstgen.dta", replace
