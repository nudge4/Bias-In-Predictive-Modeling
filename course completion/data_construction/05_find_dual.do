clear
forvalues y=2014/2019 {
	foreach v in 3_spe 4_sue 6_fae {
		preserve
			zipuse "D:\Yifeng -- Project Work\ys8mz_sandbox\Build\Student\Student_`y'_`v'.dta.zip", clear
			keep vccsid strm dual_enrollment
			bys vccsid: egen dual_ind = max(dual_enrollment)
			drop dual_enrollment
			duplicates drop
			isid vccsid
			sort vccsid
			tempfile dual
			save "`dual'", replace
		restore
		append using "`dual'", force
	}
}
sort vccsid strm
save "C:\\Users\\ys8mz\\Box Sync\\Clickstream\\data\\bias\\dual.dta", replace


clear
forvalues y=2000/2019 {
	foreach v in 3_spe 4_sue 6_fae {
		if "`y'_`v'" != "2000_3_spe" {
			preserve
				zipuse "D:\Yifeng -- Project Work\ys8mz_sandbox\Build\Student\Student_`y'_`v'.dta.zip", clear
				keep vccsid strm dual_enrollment
				bys vccsid: egen dual_ind = max(dual_enrollment)
				drop dual_enrollment
				duplicates drop
				isid vccsid
				sort vccsid
				tempfile dual
				save "`dual'", replace
			restore
			append using "`dual'", force
		}
	}
}
sort vccsid strm

preserve
	use "C:\Users\ys8mz\Box Sync\Clickstream\data\bias\term_lvl_gpa_enrl_intensity.dta", clear
	keep vccsid strm
	duplicates drop
	tempfile all_id_strm
	save "`all_id_strm'", replace
restore
merge 1:1 vccsid strm using "`all_id_strm'", keep(2 3) nogen
replace dual_ind = 0 if mi(dual_ind)
sort vccsid strm


foreach t in 2142 2143 2144 2152 2153 2154 2162 2163 2164 2172 2173 2174 2182 2183 2184 2192 2193 2194 {
	preserve
		keep if strm < `t'
		egen ever_dual = max(dual_ind), by(vccsid)
		gen target_term = `t'
		drop dual_ind strm
		duplicates drop
		tempfile ever_`t'
		save "`ever_`t''", replace
	restore
}
clear
foreach t in 2142 2143 2144 2152 2153 2154 2162 2163 2164 2172 2173 2174 2182 2183 2184 2192 2193 2194 {
	append using "`ever_`t''", force
}
sort vccsid target_term
rename target_term strm
merge 1:1 vccsid strm using "C:\\Users\\ys8mz\\Box Sync\\Clickstream\\data\\bias\\all_vccsid_strm.dta", keep(2 3) nogen
drop first_strm
save "C:\\Users\\ys8mz\\Box Sync\\Clickstream\\data\\bias\\ever_dually_enrolled.dta", replace
