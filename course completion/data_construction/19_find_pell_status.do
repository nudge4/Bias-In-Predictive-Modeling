use "D:\Yifeng -- Project Work\ys8mz_sandbox\bias_analyses\all_vccsid_strm.dta", clear
drop strm
duplicates drop
sort vccsid
forvalues y=214/219 {
	foreach t in 2 3 4 {
		local strm = `y'*10+`t'
		gen strm_`strm' = 1
	}
}
reshape long strm_@, i(vccsid) j(term)
drop strm_
rename term strm
sort vccsid strm

merge 1:1 vccsid strm using "D:\Yifeng -- Project Work\ys8mz_sandbox\bias_analyses\pell_by_term.dta", keep(1 3) nogen
sort vccsid strm
forvalues y=214/219 {
	foreach t in 2 3 4 {
		local strm = `y'*10+`t'
		if `strm' != 2072 {
			preserve
				keep if strm <= `strm'
				egen max_pell = max(pell), by(vccsid)
				keep if strm == `strm'
				tempfile pell_`strm'
				save "`pell_`strm''", replace
			restore
		}
	}
}

clear
forvalues y=214/219 {
	foreach t in 2 3 4 {
		local strm = `y'*10+`t'
		if `strm' != 2072 {
			append using "`pell_`strm''", force
		}
	}
}

sort vccsid strm
drop pell

merge 1:1 vccsid strm using "D:\Yifeng -- Project Work\ys8mz_sandbox\bias_analyses\all_vccsid_strm.dta", keep(2 3) nogen
sort vccsid strm
gen pell_ever_0 = 0
gen pell_ever_1 = 0
replace pell_ever_0 = 1 if max_pell == 1
replace pell_ever_1 = 1 if max_pell == 0
drop max_pell

merge 1:1 vccsid strm using "D:\Yifeng -- Project Work\ys8mz_sandbox\bias_analyses\pell_by_term.dta", keep(1 3) nogen
gen pell_target_0 = 0
gen pell_Target_1 = 0
replace pell_target_0 = 1 if pell == 0
replace pell_target_1 = 1 if pell == 1
drop pell

save "D:\Yifeng -- Project Work\ys8mz_sandbox\bias_analyses\pell.dta", replace
