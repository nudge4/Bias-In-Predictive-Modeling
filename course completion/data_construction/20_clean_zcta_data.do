use "C:\Users\ys8mz\Downloads\zcta_poverty_median_income_enlarged.dta", clear
preserve
	use "C:\Users\ys8mz\Downloads\zcta_poverty_median_income.dta", clear
	drop if mi(year)
	gen perc_below_pov_mi_type = mi(perc_below_pov)
	gen median_income_mi_type = mi(median_income)
	rename perc_below_pov perc_below_pov_orig
	rename median_income_households median_income_orig
	keep zcta year perc_below_pov_mi_type median_income_mi_type perc_below_pov_orig median_income_orig
	tempfile tt
	save "`tt'", replace
restore
merge 1:1 zcta year using "`tt'", keep(1 3) nogen
replace perc_below_pov_mi_type = 1 if !mi(perc_below_pov) & mi(perc_below_pov_mi_type)
replace median_income_mi_type = 1 if !mi(median_income_households) & mi(median_income_mi_type)
replace perc_below_pov_mi_type = 2 if mi(perc_below_pov)
replace median_income_mi_type = 2 if mi(median_income_households)
save "C:\Users\ys8mz\Downloads\zcta_poverty_median_income_enlarged_2.dta", replace
