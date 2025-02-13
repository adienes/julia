# This file is a part of Julia. License is MIT: https://julialang.org/license

module Error

import ..LibGit2: ensure_initialized
using LibGit2_jll

export GitError

@enum(Code, GIT_OK          = Cint(0),   # no error
            ERROR           = Cint(-01), # generic error
            ENOTFOUND       = Cint(-03), # requested object could not be found
            EEXISTS         = Cint(-04), # object exits preventing op
            EAMBIGUOUS      = Cint(-05), # more than one object matches
            EBUFS           = Cint(-06), # output buffer too small to hold data
            EUSER           = Cint(-07), # user callback generated error
            EBAREREPO       = Cint(-08), # operation not allowed on bare repo
            EUNBORNBRANCH   = Cint(-09), # HEAD refers to branch with 0 commits
            EUNMERGED       = Cint(-10), # merge in progress prevented op
            ENONFASTFORWARD = Cint(-11), # ref not fast-forwardable
            EINVALIDSPEC    = Cint(-12), # name / ref not in valid format
            ECONFLICT       = Cint(-13), # Checkout conflicts prevented operation
            ELOCKED         = Cint(-14), # lock file prevented op
            EMODIFIED       = Cint(-15), # ref value does not match expected
            EAUTH           = Cint(-16), # authentication error
            ECERTIFICATE    = Cint(-17), # server certificate is invalid
            EAPPLIED        = Cint(-18), # patch/merge has already been applied
            EPEEL           = Cint(-19), # the requested peel operation is not possible
            EEOF            = Cint(-20), # unexpected EOF
            EINVALID        = Cint(-21), # Invalid operation or input
            EUNCOMMITTED    = Cint(-22), # Uncommitted changes in index prevented operation
            EDIRECTORY      = Cint(-23), # The operation is not valid for a directory
            EMERGECONFLICT  = Cint(-24), # A merge conflict exists and cannot continue

            PASSTHROUGH     = Cint(-30), # internal only
            ITEROVER        = Cint(-31), # signals end of iteration
            RETRY           = Cint(-32), # internal only
            EMISMATCH       = Cint(-33), # hashsum mismatch in object
            EINDEXDIRTY     = Cint(-34), # unsaved changes in the index would be overwritten
            EAPPLYFAIL      = Cint(-35), # patch application failed
            EOWNER          = Cint(-36), # the object is not owned by the current user
            TIMEOUT         = Cint(-37), # The operation timed out
            EUNCHANGED      = Cint(-38), # There were no changes
            ENOTSUPPORTED   = Cint(-39), # An option is not supported
            EREADONLY       = Cint(-40), # The subject is read-only
)

@enum(Class, None,
             NoMemory,
             OS,
             Invalid,
             Reference,
             Zlib,
             Repository,
             Config,
             Regex,
             Odb,
             Index,
             Object,
             Net,
             Tag,
             Tree,
             Indexer,
             SSL,
             Submodule,
             Thread,
             Stash,
             Checkout,
             FetchHead,
             Merge,
             SSH,
             Filter,
             Revert,
             Callback,
             CherryPick,
             Describe,
             Rebase,
             Filesystem,
             Patch,
             WorkTree,
             SHA1,
             HTTP,
             Internal,
             Grafts)

struct ErrorStruct
    message::Ptr{UInt8}
    class::Cint
end

struct GitError <: Exception
    class::Class
    code::Code
    msg::String
end
Base.show(io::IO, err::GitError) = print(io, "GitError(Code:$(err.code), Class:$(err.class), $(err.msg))")

function last_error()
    ensure_initialized()
    err = ccall((:git_error_last, libgit2), Ptr{ErrorStruct}, ())
    if err != C_NULL
        err_obj   = unsafe_load(err)
        err_class = Class(err_obj.class)
        err_msg   = unsafe_string(err_obj.message)
    else
        err_class = Class(0)
        err_msg = "No errors"
    end
    return (err_class, err_msg)
end

GitError(err_code::Integer) = GitError(Code(err_code))
function GitError(err_code::Code)
    err_class, err_msg = last_error()
    return GitError(err_class, err_code, err_msg)
end

end # Error module

macro check(git_func)
    quote
        err = Cint($(esc(git_func::Expr)))
        if err < 0
            throw(Error.GitError(err))
        end
        err
    end
end
