if !isdefined(@__MODULE__, :test_log)
    const TEST_DEBUG = lowercase(get(ENV, "TEST_DEBUG", "false")) in ("1", "true", "yes", "on")
    test_log(args...) = TEST_DEBUG ? println(args...) : nothing
end