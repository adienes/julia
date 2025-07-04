#include <llvm/ADT/StringMap.h>
#include <llvm/DebugInfo/DIContext.h>
#include <llvm/IR/DataLayout.h>

#include "julia.h"

#include <map>
#include <mutex>
#include <type_traits>

typedef struct {
    const llvm::object::ObjectFile *obj;
    llvm::DIContext *ctx;
    int64_t slide;
    std::map<uintptr_t, StringRef, std::greater<size_t>> *symbolmap;
} jl_object_file_entry_t;

// Central registry for resolving function addresses to `jl_code_instance_t`s and
// originating `ObjectFile`s (for the DWARF debug info).
//
// A global singleton instance is notified by the JIT whenever a new object is emitted,
// and later queried by the various function info APIs. We also use the chance to handle
// some platform-specific unwind info registration (which is unrelated to the query
// functionality).
class JITDebugInfoRegistry
{
public:
    template<typename ResourceT>
    struct Locked {

        template<typename CResourceT>
        struct Lock {
            std::unique_lock<std::mutex> lock;
            CResourceT &resource;

            Lock(std::mutex &mutex, CResourceT &resource) JL_NOTSAFEPOINT JL_NOTSAFEPOINT_ENTER : lock(mutex), resource(resource) {}
            Lock(Lock &&) JL_NOTSAFEPOINT = default;
            Lock &operator=(Lock &&) JL_NOTSAFEPOINT = default;

            CResourceT &operator*() JL_NOTSAFEPOINT {
                return resource;
            }

            const CResourceT &operator*() const JL_NOTSAFEPOINT {
                return resource;
            }

            CResourceT *operator->() JL_NOTSAFEPOINT {
                return &**this;
            }

            const CResourceT *operator->() const JL_NOTSAFEPOINT {
                return &**this;
            }

            operator const CResourceT &() const JL_NOTSAFEPOINT {
                return resource;
            }

            ~Lock() JL_NOTSAFEPOINT JL_NOTSAFEPOINT_LEAVE = default;
        };
    private:

        mutable std::mutex mutex;
        ResourceT resource;
    public:
        typedef Lock<ResourceT> LockT;
        typedef Lock<const ResourceT> ConstLockT;

        Locked(ResourceT resource = ResourceT()) JL_NOTSAFEPOINT : mutex(), resource(std::move(resource)) {}

        LockT operator*() JL_NOTSAFEPOINT JL_NOTSAFEPOINT_ENTER {
            return LockT(mutex, resource);
        }

        ConstLockT operator*() const JL_NOTSAFEPOINT JL_NOTSAFEPOINT_ENTER {
            return ConstLockT(mutex, resource);
        }

        ~Locked() JL_NOTSAFEPOINT JL_NOTSAFEPOINT_LEAVE = default;
    };

    struct image_info_t {
        uint64_t base;
        jl_image_fptrs_t fptrs;
        jl_code_instance_t **fvars_cinst;
        size_t fvars_n;
    };

    struct libc_frames_t {
#if defined(_OS_DARWIN_) && defined(LLVM_SHLIB)
        typedef void (*frame_register_func)(void *) JL_NOTSAFEPOINT;
        std::atomic<frame_register_func> libc_register_frame_{nullptr};
        std::atomic<frame_register_func> libc_deregister_frame_{nullptr};

        void libc_register_frame(const char *Entry) JL_NOTSAFEPOINT;

        void libc_deregister_frame(const char *Entry) JL_NOTSAFEPOINT;
#endif
    };
private:

    struct LazyObjectInfo {
        SmallVector<uint8_t, 0> data;
        size_t uncompressedsize;
        std::unique_ptr<const llvm::object::ObjectFile> object;
        std::unique_ptr<llvm::DIContext> context;
        LazyObjectInfo() = delete;
        ~LazyObjectInfo() JL_NOTSAFEPOINT = default;
    };

    struct SectionInfo {
        LazyObjectInfo *object;
        size_t SectionSize;
        ptrdiff_t slide;
        uint64_t SectionIndex;
        SectionInfo() = delete;
        ~SectionInfo() JL_NOTSAFEPOINT = default;
    };

    template<typename KeyT, typename ValT>
    using rev_map = std::map<KeyT, ValT, std::greater<KeyT>>;

    typedef rev_map<size_t, SectionInfo> objectmap_t;
    typedef rev_map<uint64_t, jl_object_file_entry_t> objfilemap_t;

    objectmap_t objectmap{};
    rev_map<size_t, std::pair<size_t, jl_code_instance_t *>> cimap{};

    // Maintain a mapping of unrealized function names -> linfo objects
    // so that when we see it get emitted, we can add a link back to the linfo
    // that it came from (providing name, type signature, file info, etc.)
    Locked<llvm::StringMap<jl_code_instance_t*>> codeinst_in_flight{};

    Locked<llvm::DenseMap<uint64_t, image_info_t>> image_info{};

    Locked<objfilemap_t> objfilemap{};

    static std::string mangle(llvm::StringRef Name, const llvm::DataLayout &DL) JL_NOTSAFEPOINT;

public:

    JITDebugInfoRegistry() JL_NOTSAFEPOINT;
    ~JITDebugInfoRegistry() JL_NOTSAFEPOINT = default;

    libc_frames_t libc_frames{};

    void add_code_in_flight(llvm::StringRef name, jl_code_instance_t *codeinst, const llvm::DataLayout &DL) JL_NOTSAFEPOINT;
    jl_code_instance_t *lookupCodeInstance(size_t pointer) JL_NOTSAFEPOINT;
    void registerJITObject(const llvm::object::ObjectFile &Object,
                        std::function<uint64_t(const llvm::StringRef &)> getLoadAddress) JL_NOTSAFEPOINT;
    objectmap_t& getObjectMap() JL_NOTSAFEPOINT;
    void add_image_info(image_info_t info) JL_NOTSAFEPOINT;
    bool get_image_info(uint64_t base, image_info_t *info) const JL_NOTSAFEPOINT;
    Locked<objfilemap_t>::LockT get_objfile_map() JL_NOTSAFEPOINT;

    std::shared_mutex symbol_mutex;
};
