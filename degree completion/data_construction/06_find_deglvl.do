/* This script identifies the degree level each student is pursuing during each enrolled semester. This information will be used in analyses that have to do with degree-level-specific models. */

clear
forvalues y=2007/2017 {
	foreach t in "3_spe" "4_sue" "6_fae" {
		preserve
			zipuse "D:\Yifeng -- Project Work\ys8mz_sandbox\Build\Student\Student_`y'_`t'.dta.zip", clear
			replace acadplan_deglvl = "OTHER" if !inlist(acadplan_deglvl, "AA&S", "AAS", "CERT", "CSC")
			keep vccsid strm college acadplan_deglvl total_credit_hrs
			gsort vccsid acadplan_deglvl -total_credit_hrs
			bys vccsid: keep if _n == 1
			keep vccsid strm acadplan_deglvl
			rename acadplan_deglvl deglvl
			isid vccsid
			sort vccsid
			tempfile stu
			save "`stu'", replace
		restore
		append using "`stu'", force
	}
}
drop if strm == 2072
rename strm last_term
sort vccsid last_term
isid vccsid last_term

merge 1:1 vccsid last_term using "C:\Users\ys8mz\Box Sync\Predictive Models of College Completion (VCCS)\intermediate_files\new\full_data_truncated.dta", keepusing(first_nonde_strm) keep(2 3) nogen
drop first_nonde_strm
replace deglvl = "OTHER" if mi(deglvl)

drop last_term
sort vccsid
save "C:\Users\ys8mz\Box Sync\Predictive Models of College Completion (VCCS)\intermediate_files\new\bias\student_deglvl.dta", replace
