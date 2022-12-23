module Analysis
using PPredict, Plots, JLD2, Dierckx
export read_data, split_data, plot_by_trt, split_shift

####################################################################
# colors, see MMAColors.jl in my private modules

mma = [RGB(0.3684,0.50678,0.7098),RGB(0.8807,0.61104,0.14204),
			RGB(0.56018,0.69157,0.19489), RGB(0.92253,0.38563,0.20918)];

####################################################################

# data rows: N, F, res_size, shift, dbl, r2_tr, r2_test
# each column from one run, here favoring column major approach of julia
function read_data()
	path = "/Users/steve/sim/zzOtherLang/julia/projects/Precise/PPredict/output"
	jld_files = filter(x->occursin("jld2",x), readdir(path,join=true))
	num_cols = size(jld_files)[1]
	data = zeros(7, num_cols)
	for (f,j) in zip(jld_files,1:num_cols)
		jd = load(f)
		jdd = jd["exp_data"]
		data[1,j] = jdd.S.N
		data[2,j] = jdd.S.F
		data[3,j] = jdd.S.res_size
		data[4,j] = jdd.S.shift
		st = calc_stats(jdd.S, jdd.mach, jdd.target_train, jdd.y_train,
							jdd.target_test, jdd.y_test)
		dbl = abs(log(2)/st.ly)
		data[5,j] = dbl > 5.0 ? 5.0 : dbl
		data[6,j] = st.r2_train
		data[7,j] = st.r2_test
	end
	# keep columns for which dbl value in 5th row is < 2, otherwise drop
	return data[:,data[5,:] .<= 2.0]
end

# pick out columns based on row value, here N in first row, d[:,d[1,:] .== 5.0]
# split by N and res_size, using dbl as x axis for plots, r2_test as y axis
# separate plots for each N, separate curves for each res_size
# container is a matrix of matrices, rows for unique N, cols for unique res_size

function split_data(d=nothing)
	if d == nothing d = read_data() end
	N_val = sort(unique(d[1,:]))
	res_val = sort(unique(d[3,:]))
	N_num = length(N_val)
	res_num = length(res_val)
	by_trt = Matrix{Matrix{Float64}}(undef,N_num,res_num)
	for (N,i) in zip(N_val,1:N_num)
		N_tmp = d[:,d[1,:] .== N]
		for (res,j) in zip(res_val,1:res_num)
			tmp = by_trt[i,j] = N_tmp[:,N_tmp[3,:] .== res]
			tmp .= tmp[:,sortperm(tmp[5,:])]
		end
	end
	return by_trt
end

# split matrix by shift in 4th row
function split_shift(trt)
	shift_val = sort(unique(trt[4,:]))
	shift_num = length(shift_val)
	by_shift = Array{Matrix{Float64}}(undef,shift_num)
	for (shift,i) in zip(shift_val,1:shift_num)
		tmp = by_shift[i] = trt[:,trt[4,:] .== shift]
		tmp .= tmp[:,sortperm(tmp[5,:])]
	end
	return by_shift
end

# N in cols, res_size in rows, shift in separate curve per plot
# each y vs x curve is r2_tst vs dbl
function plot_by_trt(d=nothing; show_points=false)
	by_trt = split_data(d)
	N_num = size(by_trt)[1]
	res_num = size(by_trt)[2]	
	pl = plot(layout=(res_num,N_num), size=(800*N_num,500*res_num), legend=:none)
	for i in 1:N_num
		for j in 1:res_num
			by_shift = split_shift(by_trt[i,j])
			for (s,k) in zip(by_shift,1:length(by_shift))
				mn = minimum(s[5,:])
				mx = maximum(s[5,:])
				step = (mx-mn)/8.0
				rng = mn:step:mx
				interp = Spline1D(s[5,:],s[7,:];k=1,s=0.0)
				if show_points scatter!(s[5,:],s[7,:],subplot=1 + (i-1) + (j-1)*N_num,
						color=mma[k]) end
				plot!(rng,interp.(rng),subplot=1 + (i-1) + (j-1)*N_num,
						color=mma[k], linewidth=2)
			end
		end
	end
	display(pl)
	return pl
end

end # module Analysis
