# Rebuild the article's six-panel figure from the audited ADD results.
ENV["GKSwstype"]="100"
include("report_ADD_tuning.jl")
include("plot_ADD_tuning.jl")
report=analyze_add()
plot_add_article(report)
println("Saved ADD_selected_strategies.pdf and .png")
