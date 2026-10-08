using Plots,DelimitedFiles,Statistics
gr()
root=normpath(joinpath(@__DIR__,"..",".."))
input=joinpath(root,"data","protes_comparison","article_plots")
out=joinpath(root,"examples","pictures");mkpath(out)
default(fontfamily="Computer Modern",linewidth=2,markersize=4,
        guidefontsize=11,tickfontsize=9,legendfontsize=8,titlefontsize=12,
        framestyle=:box,gridalpha=0.2,background_color=:white)
colors=palette(:default)
readtable(name)=readdlm(joinpath(input,name),',',Any;skipstart=1)
function save(p,name)
    savefig(p,joinpath(out,name*".pdf"));savefig(p,joinpath(out,name*".png"))
end
C(n,k)=0<=k<=n ? binomial(big(n),k) : big(0)
function rank(n,k)
    k in (0,n*n) && return big(1)
    r,s=divrem(k,n)
    C(n,r)-C(s,r-n+s)+C(n,r+1)-C(n-s,r+1)
end
blocks(n)=sum(C(n-1,r)+C(n,r)-C(s+1,r-n+s+1)+C(n,r+1)-C(n-s,r+1) for r in 0:n-1 for s in 0:n-1)
ns=2:20
fast=[Float64(8*(2*(n*n-2)*maximum(rank(n,k) for k in 0:n*n)^2+4*maximum(rank(n,k) for k in 0:n*n)))/2^20 for n in ns]
general=[Float64(16*sum(rank(n,k-1)*rank(n,k) for k in 1:n*n))/2^20 for n in ns]
p=plot(ns,fast;label="PROTES: uniform dense rank",color=colors[1],yscale=:log10,
       xlabel="Assignment size n",ylabel="Storage (MiB)",xticks=2:2:20,yticks=10.0.^(-4:2:10),size=(500,360),legend=:topleft)
plot!(p,ns,general;label="PROTES general: variable dense ranks",color=colors[2])
plot!(p,ns,[Float64(8*blocks(n))/2^20 for n in ns];label="U(1): scalar blocks only",color=colors[3])
mem=readtable("memory.csv")
plot!(p,Int.(mem[:,1]),Float64.(mem[:,2])./2^20;label="U(1): measured object incl. metadata",color=colors[4],marker=:circle,linestyle=:dash)
save(p,"benchmark_protes_memory_julia")
speed=readtable("speed.csv")
p=plot(;xlabel="Assignment size n",ylabel="Samples per second",yscale=:log10,xticks=4:2:16,size=(500,360),legend=:topright)
for (j,(method,label)) in enumerate([("U1","U(1)"),("fast","PROTES: uniform dense rank"),("general","PROTES general")])
    rows=speed[speed[:,1].==method,:];sizes=sort(unique(Int.(rows[:,2])))
    med=[median(Float64.(rows[rows[:,2].==n,4])) for n in sizes]
    low=[minimum(Float64.(rows[rows[:,2].==n,4])) for n in sizes]
    high=[maximum(Float64.(rows[rows[:,2].==n,4])) for n in sizes]
    plot!(p,sizes,med;label,color=colors[j],marker=:circle,ribbon=(med.-low,high.-med),fillalpha=0.15)
end
save(p,"benchmark_protes_sampling_julia")
data=readtable("novelty.csv")
names=["default","rank20","slow10","batch400","fit20","U1 EELS"]
ticks=[0,1,10,100,1000,10000];ytransform(x)=log10(1+x)
for budget in [5,50]
    panels=[]
    for obj in ["linear","quadratic"]
        p=plot(;title=uppercasefirst(obj),ylabel="New distinct feasible points",yticks=(ytransform.(ticks),string.(ticks)),
               ylims=(-0.06,4.15),legend=budget==5 ? :outerbottom : :topleft,legend_columns=budget==5 ? 3 : 1,
               bottom_margin=7Plots.mm,left_margin=4Plots.mm)
        for (j,name) in enumerate(names)
            rows=data[(data[:,1].==name).&(data[:,2].==obj).&(data[:,5].==budget),:]
            if budget==5
                rows=rows[sortperm(Int.(rows[:,3])),:]
                plot!(p,Int.(rows[:,3]),ytransform.(Float64.(rows[:,6]));label=name,color=colors[j],marker=:circle)
            else
                vals=Float64.(rows[:,6]);x=j .+ collect(range(-0.13,0.13,length=length(vals)))
                scatter!(p,x,ytransform.(vals);label=false,color=colors[j],markerstrokewidth=0)
                plot!(p,[j-.22,j+.22],fill(ytransform(mean(vals)),2);color=:darkorange,linewidth=3,label=j==1 ? "Group average" : false)
                scatter!(p,[j],[ytransform(median(vals))];color=:red,marker=:diamond,markersize=4,label=j==1 ? "Group median" : false)
            end
        end
        budget==5 ? plot!(p;title=uppercasefirst(obj)*" (assignment size n)",xticks=4:10,xlims=(3.8,10.2)) : plot!(p;xticks=(1:6,names),xrotation=25,xlims=(.6,6.4))
        push!(panels,p)
    end
    save(plot(panels...;layout=(1,2),size=(1100,440)),"benchmark_protes_u1_$(budget)s_julia")
end
println("Saved four article figures (six panels), rendered entirely with Julia/Plots.")
