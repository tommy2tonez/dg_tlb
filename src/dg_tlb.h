#ifndef __DG_TLB_H__
#define __DG_TLB_H__

#include <vector>
#include <stdint.h>
#include <stdlib.h>
#include <memory>
#include <unordered_map>
#include <unordered_set>
#include <type_traits>
#include <atomic>
#include <optional>
#include <iterator>

namespace dg::flush_on_cap_tlb{
    
    using virtual_page_state_t                                                  = size_t; 
    static inline constexpr size_t PAGE_SZ                                      = size_t{1} << 20;
    static inline constexpr size_t CACHE_LINE_SIZE                              = size_t{1} << 6; 
    static inline constexpr virtual_page_state_t virtual_page_null_state        = 0u;
    static inline constexpr virtual_page_state_t virtual_page_transfer_state    = ~virtual_page_null_state; //single ownership (test_and_set-liked property), denotes buffer transfering 
    static inline constexpr size_t ID_BITCOUNT                                  = sizeof(uint16_t) * CHAR_BIT;
    static inline constexpr size_t REF_BITCOUNT                                 = sizeof(uint16_t) * CHAR_BIT; 

    static_assert(ID_BITCOUNT + REF_BITCOUNT <= sizeof(virtual_page_state_t) * CHAR_BIT);

    using mem_transfer_device_t = void (*) (void *, const void *, size_t) noexcept;  

    #if defined(__IS_CUDA__)

    static inline constexpr size_t MAX_ATOMIC_LOAD_SIZE = sizeof(size_t);
    using dg_atomic_flag_type = unsigned int;
    static inline constexpr unsigned int dg_atomic_flag_default_value = 0u; 

    template <class T>
    using dg_atomic_type = T; 

    template <class T, std::enable_if_t<std::is_fundamental_v<T> && sizeof(T) <= MAX_ATOMIC_LOAD_SIZE, bool> = true>
    __device__ inline auto dg_atomic_load(const T& obj, std::memory_order) noexcept -> T{

        return obj;
    }

    __device__ inline auto dg_atomic_fetch_add(int& obj, int val, const std::memory_order) noexcept -> int{

        return atomicAdd(&obj, val);
    }

    __device__ inline auto dg_atomic_fetch_add(unsigned int& obj, unsigned int val, const std::memory_order) noexcept -> unsigned int{

        return atomicAdd(&obj, val);
    } 

    __device__ inline auto dg_atomic_fetch_add(unsigned long long int& obj, unsigned long long int val, const std::memory_order) noexcept -> unsigned long long int{

        return atomicAdd(&obj, val);
    }

    __device__ inline auto dg_atomic_fetch_add(float& obj, float val, const std::memory_order) noexcept -> float{

        return atomicAdd(&obj, val);
    }

    __device__ inline auto dg_atomic_fetch_add(double& obj, double val, const std::memory_order) noexcept -> double{

        return atomicAdd(&obj, val);
    }

    __device__ inline auto dg_atomic_fetch_sub(int& obj, int val, const std::memory_order) noexcept -> int{

        return atomicSub(&obj, val);
    }

    __device__ inline auto dg_atomic_fetch_sub(unsigned int& obj, unsigned int val, const std::memory_order) noexcept -> unsigned int{

        return atomicSub(&obj, val);
    }

    __device__ inline auto dg_atomic_flag_test_and_set(unsigned int& obj, const std::memory_order) noexcept -> unsigned int{

        return atomicExch(&obj, unsigned_int{1u}) == 0u; //cmp to 0u is faster due to nullptr optimization -
    }

    __device__ inline void dg_atomic_flag_clear(unsigned int& obj, const std::memory_order) noexcept{

        atomicExch(&obj, unsigned_int{0u});
    }

    __device__ inline auto dg_atomic_exchange(int& obj, int val, const std::memory_order) noexcept -> int{

        return atomicExch(&obj, val);
    }

    __device__ inline auto dg_atomic_exchange(unsigned int& obj, unsigned int val, const std::memory_order) noexcept -> unsigned int{

        return atomicExch(&obj, val);
    }

    __device__ inline auto dg_atomic_exchange(unsigned long long int& obj, unsigned long long int val, const std::memory_order) noexcept -> unsigned long long int{

        return atomicExch(&obj, val);
    }   

    __device__ inline auto dg_atomic_exchange(float& obj, float val, const std::memory_order) noexcept -> float{

        return atomicExch(&obj, val);
    }

    __device__ inline auto dg_compare_exchange_strong(int& obj, int cmp, int val, const std::memory_order) noexcept -> bool{

        return atomicCAS(&obj, cmp, val) == cmp;
    } 

    __device__ inline auto dg_compare_exchange_strong(unsigned int& obj, unsigned int cmp, unsigned int val, const std::memory_order) noexcept -> bool{

        return atomicCAS(&obj, cmp, val) == cmp;
    }

    __device__ inline auto dg_compare_exchange_strong(unsigned long long int& obj, unsigned long long int cmp, unsigned long long int val, const std::memory_order) noexcept -> bool{

        return atomicCAS(&obj, cmp, val) == cmp;
    }

    __device__ inline auto dg_compare_exchange_strong(unsigned short int& obj, unsigned short int cmp, unsigned short int val, const std::memory_order) noexcept -> bool{

        return atomicCAS(&obj, cmp, val) == cmp;
    }

    __device__ inline void dg_atomic_thread_fence(const std::memory_order mem_order) noexcept{

        (void) mem_order;
    }

    #else

    using dg_atomic_flag_type = std::atomic_flag;
    
    template <class T>
    using dg_atomic_type = std::atomic<T>; 

    static inline constexpr bool dg_atomic_flag_default_value = false;

    template <class T>
    inline auto dg_atomic_load(const std::atomic<T>& obj, const std::memory_order mem_order) noexcept -> T{

        return obj.load(mem_order);
    }

    template <class T>
    inline auto dg_atomic_fetch_add(std::atomic<T>& obj, T val, const std::memory_order mem_order) noexcept -> T{

        return obj.fetch_add(val, mem_order);
    }

    template <class T>
    inline auto dg_atomic_fetch_sub(std::atomic<T>& obj, T val, const std::memory_order mem_order) noexcept -> T{

        return obj.fetch_sub(val, mem_order);
    }

    inline auto dg_atomic_flag_test_and_set(std::atomic_flag& obj, const std::memory_order mem_order) noexcept -> bool{

        return obj.test_and_set(mem_order);
    }

    inline void dg_atomic_flag_clear(std::atomic_flag& obj, const std::memory_order mem_order) noexcept{

        obj.clear(mem_order);
    }

    template <class T>
    inline auto dg_atomic_exchange(std::atomic<T>& obj, T val, const std::memory_order mem_order) noexcept -> T{

        return obj.exchange(val, mem_order);
    }

    template <class T>
    inline auto dg_compare_exchange_strong(std::atomic<T>& obj, T cmp, T val, const std::memory_order mem_order) noexcept -> bool{

        return obj.compare_exchange_strong(cmp, val, mem_order);
    } 

    inline void dg_atomic_thread_fence(const std::memory_order mem_order) noexcept{

        std::atomic_thread_fence(mem_order);
    }

    #endif 

    struct no_page_found: std::exception{}; 

    struct Config{
        void * translator_addr; //should be origined from char * (avoid UB - pointer arithmetic on std-qualified char array - whose pointer is obtained from new[] operation)
        size_t translator_sz;
        void * translatee_addr; //should be origined from char * (avoid UB - pointer arithmetic on std-qualified char array - whose pointer is obtained from new[] operation) 
        size_t translatee_sz;
        mem_transfer_device_t virtual_to_physical_transfer_device;
        mem_transfer_device_t physical_to_virtual_transfer_device;
    };

    struct PhysicalPageState{
        alignas(CACHE_LINE_SIZE) void * addr; //immutable void * const  
        alignas(CACHE_LINE_SIZE) dg_atomic_flag_type is_acquired; //true denotes linkage to at most 1 page, false denotes no linkage + mem-safe (addr is not referenced by any at-the-time variables)
    };

    //rules:
    //null_state denotes no linkage to any physical page + virtual memory spanned by the page is up-to-date + mem-safe (addr is not referenced by any at-the-time variables). 
    //transfer_state denotes single ownership (atomic_flag, true == virtual_page_transfer_state, false otherwise), buffer transferring in progress
    //other state denotes the at-the-time physical page linkage + reference counting 
    //physical -> virtual is "injective" 
    //an atomic change to the state is valid if all reqs are met 
    struct VirtualPageState{
        alignas(CACHE_LINE_SIZE) dg_atomic_type<virtual_page_state_t> state; 
    };

    struct Table{
        Config config; 
        PhysicalPageState * physical_page_list;
        size_t physical_page_list_sz;
        VirtualPageState * virtual_page_list;
        size_t virtual_page_list_sz;
    };

    template <class T, size_t BIT_COUNT, std::enable_if_t<std::is_unsigned_v<T>, bool> = true>
    constexpr inline auto low(const std::integral_constant<size_t, BIT_COUNT>) noexcept -> T{

        static_assert(BIT_COUNT <= sizeof(T) * CHAR_BIT);

        if constexpr(BIT_COUNT == sizeof(T) * CHAR_BIT){
            return ~T{0u};
        } else{
            return (T{1u} << BIT_COUNT) - 1;
        }
    }

    //consider integral_constant, gcc optimizes const <arithmetic_type> (1 prop) if inline but no guarantee that would happen
    constexpr inline auto slot(const size_t idx, const size_t page_sz) noexcept -> size_t{

        return idx / page_sz;
    }

    constexpr inline auto offset(const size_t idx, const size_t page_sz) noexcept -> size_t{ 

        return idx % page_sz;
    } 

    constexpr inline auto size(const size_t idx, const size_t page_sz) noexcept -> size_t{

        return slot(idx, page_sz) + 1;
    }

    constexpr inline auto index(const size_t _slot, const size_t page_sz) noexcept -> size_t{

        return _slot * page_sz;
    } 

    constexpr inline auto index(const size_t _slot, const size_t _offset, const size_t page_sz) noexcept -> size_t{

        return index(_slot, page_sz) + _offset;
    } 

    static inline Config config; //
    static inline Table table; //

    //memory-deduced-qualified == the result of (void) or (stateful) variables can be used to deduce the up-to-date of the at-the-time related variables

    //try acquire an empty_page. If found return up-to-date memory-deduced-qualified page_idx, nullopt otherwise (optional memory-deduced-qualified)
    inline auto physical_page_try_acquire_empty() -> std::optional<size_t>{

        for (size_t i = 0; i < table.physical_page_list_sz; ++i){
            if (dg_atomic_flag_test_and_set(table.physical_page_list[i].is_acquired, std::memory_order_acq_rel)){
                return i;
            }
        }

        return std::nullopt;
    }

    //release page_idx(not memory-deduced-qualified)
    inline void physical_page_release(size_t page_idx) noexcept{

        dg_atomic_flag_clear(table.physical_page_list[page_idx].is_acquired, std::memory_order_release);
    } 

    //make virtual_page state from page_idx + counter. a valid page_state is not null_state or transfer_state
    constexpr auto virtual_page_make(size_t page_idx, size_t counter) noexcept -> virtual_page_state_t{

        return (static_cast<virtual_page_state_t>(page_idx + 1) << REF_BITCOUNT) | static_cast<virtual_page_state_t>(counter);
    } 

    //extract idx from a valid page_state
    constexpr auto virtual_page_extract_idx(virtual_page_state_t state) noexcept -> size_t{

        return (state >> REF_BITCOUNT) - 1;
    } 

    //extract counter from a valid page_state
    constexpr auto virtual_page_extract_counter(virtual_page_state_t state) noexcept -> size_t{

        return state & low<virtual_page_state_t>(std::integral_constant<size_t, REF_BITCOUNT>{});
    }

    //exhaust physical_mem_space -> virtual_mem_space  
    inline void virtual_physical_page_sync(size_t virtual_page_idx, size_t physical_page_idx) noexcept{

        char * virtual_ptr  = static_cast<char *>(config.translator_addr) + index(virtual_page_idx, PAGE_SZ);
        char * physical_ptr = static_cast<char *>(config.translatee_addr) + index(physical_page_idx, PAGE_SZ);

        config.physical_to_virtual_transfer_device(virtual_ptr, static_cast<const void *>(physical_ptr), PAGE_SZ);
    } 

    //transfer virtual_mem_space -> physical_mem_space
    inline void physical_virtual_page_sync(size_t physical_page_idx, size_t virtual_page_idx) noexcept{

        char * physical_ptr = static_cast<char *>(config.translatee_addr) + index(physical_page_idx, PAGE_SZ);
        char * virtual_ptr  = static_cast<char *>(config.translator_addr) + index(virtual_page_idx, PAGE_SZ);

        config.virtual_to_physical_transfer_device(physical_ptr, static_cast<const void *>(virtual_ptr), PAGE_SZ);
    } 

    //try release a page - true if successfully released (memory-deduced-qualified, false otherwise (not memory-deduced-qualified) 
    inline auto virtual_page_try_release_if_zero_ref(size_t page_idx) noexcept -> bool{

        while (true){
            auto state = dg_atomic_load(table.virtual_page_list[page_idx].state, std::memory_order_acquire); //atomic_load as an unfair randomizer

            if (state == virtual_page_null_state){
                return true;
            }
            
            if (state == virtual_page_transfer_state){
                return false;
            }

            size_t physical_page_idx = virtual_page_extract_idx(state);
            size_t counter = virtual_page_extract_counter(state);
            
            if (counter != 0u){
                return false;
            }

            if (dg_compare_exchange_strong(table.virtual_page_list[page_idx].state, state, virtual_page_transfer_state, std::memory_order_acq_rel)){ //try acquiring atomic_flag from valid state (acq_rel is mandatory because of unfair randomizer, not mandatory otherwise (version_control))
                virtual_physical_page_sync(page_idx, physical_page_idx); //transfers memory as permission granted
                dg_atomic_exchange(table.virtual_page_list[page_idx].state, virtual_page_null_state, std::memory_order_release); //release atomic_flag + defaultize (because of injective req + single ownership, up-to-date + mem-safe req are met)
                physical_page_release(physical_page_idx); //release physical_page, mem_safe req is met (release after null_state for symmetry, as initialized)
                dg_atomic_thread_fence(std::memory_order_acquire);
                return true;
            }
        }
    }   

    //try synchronize a page - true if successfully synchronized (...), false otherwise (...). true denotes the at-the-time page is up-to-date.   
    inline auto virtual_page_try_sync(size_t page_idx) noexcept -> bool{

        while (true){
            auto state = dg_atomic_load(table.virtual_page_list[page_idx].state, std::memory_order_acquire); //atomic load as an unfair randomizer

            if (state == virtual_page_null_state){
                return true;
            }
            
            if (state == virtual_page_transfer_state){
                return false;
            }

            size_t physical_page_idx = virtual_page_extract_idx(state);
            size_t counter = virtual_page_extract_counter(state);
            
            if (counter != 0u){
                return false;
            }

            if (dg_compare_exchange_strong(table.virtual_page_list[page_idx].state, state, virtual_page_transfer_state, std::memory_order_acq_rel)){ //try acquiring atomic_flag from valid state (acq_rel is mandatory because of unfair randomizer, not mandatory otherwise (version_control))
                virtual_physical_page_sync(page_idx, physical_page_idx); //transfers memory as permission granted
                dg_atomic_exchange(table.virtual_page_list[page_idx].state, state, std::memory_order_release); //release atomic_flag + snap back to org_state (should obey the rules as others are immutable during atomic_flag acquisition - single ownership rule)
                dg_atomic_thread_fence(std::memory_order_acquire);
                return true;
            }
        } 
    } 

    //try release all available pages (not memory-deduced-qualified)
    inline void virtual_page_release_zero_ref() noexcept{

        for (size_t i = 0; i < table.virtual_page_list_sz; ++i){
            virtual_page_try_release_if_zero_ref(i);
        }
    }
    
    //try acquire an empty page - if failed - release + retry. If succeeded - return memory-deduced-qualified physical_page_idx. If throw, throw memory-deduced-qualified no_page_found 
    inline auto physical_page_force_acquire_empty() -> size_t{

        if (auto rs = physical_page_try_acquire_empty(); rs){
            return rs.value();
        }

        virtual_page_release_zero_ref();

        if (auto rs = physical_page_try_acquire_empty(); rs){
            return rs.value();
        }

        dg_atomic_thread_fence(std::memory_order_acquire);
        throw no_page_found();
    } 

    //try establish linkage to physical_page and increment reference of page_idx virtual page - return the at-the-time linked addr (memory-deduced-qualified if non-null, not qualified otherwise). The at-the-time pointer lifetime is guaranteed up-to the invoke of virtual_page_dec_ref.
    inline auto virtual_page_try_link_n_inc_ref(size_t page_idx) -> void *{

        size_t physical_page_idx = physical_page_force_acquire_empty();
        auto state  = virtual_page_make(physical_page_idx, 1u);
        physical_virtual_page_sync(physical_page_idx, page_idx);
        
        if (dg_compare_exchange_strong(table.virtual_page_list[page_idx].state, virtual_page_null_state, state, std::memory_order_acq_rel)){
            return table.physical_page_list[physical_page_idx].addr; 
        }

        physical_page_release(physical_page_idx);
        return nullptr;
    }

    //try map virtual_page to the linked physical_page + inc reference - return the at-the-time mapped addr (memory-deduced-qualified if non-null, not qualified otherwise). The at-the-time pointer lifetime is guaranteed up-to the invoke of virtual_page_dec_ref.
    inline auto virtual_page_try_map_n_inc_ref_if_exists(size_t page_idx) noexcept -> void *{

        while (true){
            auto cur_state      = dg_atomic_load(table.virtual_page_list[page_idx].state, std::memory_order_acquire);
            
            if (cur_state == virtual_page_null_state){
                return nullptr;
            }

            if (cur_state == virtual_page_transfer_state){
                continue;
            }

            size_t idx          = virtual_page_extract_idx(cur_state);
            size_t counter      = virtual_page_extract_counter(cur_state);
            auto nxt_state      = virtual_page_make(idx, counter + 1);
            
            if (dg_compare_exchange_strong(table.virtual_page_list[page_idx].state, cur_state, nxt_state, std::memory_order_acq_rel)){
                return table.physical_page_list[idx].addr;
            }
        }
    }

    //force map (link if necessary). return the non-null at-the-time mapped_addr (memory-deduced-qualified). throw no_page_found if the at-the-time linkage could not be established. The at-the-time pointer lifetime is guaranteed up-to the invoke of virtual_page_dec_ref.
    inline auto virtual_page_force_fetch_n_inc_ref(size_t page_idx) -> void *{

        while (true){
            if (void * rs = virtual_page_try_map_n_inc_ref_if_exists(page_idx); rs){
                return rs;
            }

            if (void * rs = virtual_page_try_link_n_inc_ref(page_idx); rs){
                return rs;
            }
        }
    }

    //decrease reference of the page_idx virtual_page (not memory-deduced-qualified (void))
    inline void virtual_page_dec_ref(size_t page_idx) noexcept{

        while (true){
            auto cur_state      = dg_atomic_load(table.virtual_page_list[page_idx].state, std::memory_order_acquire);

            if (cur_state == virtual_page_transfer_state){
                continue;
            }

            size_t page_idx     = virtual_page_extract_idx(cur_state);
            size_t counter      = virtual_page_extract_counter(cur_state);
            auto new_state      = virtual_page_make(page_idx, counter - 1); 

            if (dg_compare_exchange_strong(table.virtual_page_list[page_idx].state, cur_state, new_state, std::memory_order_release)){ //no guarantee that state is the same at load and cmp_exchg_strong. atomic_load as an unfair randomizer
                return;
            }
        }
    }

    //wait + drop the page_idx virtual page. the exit of this function guarantees the up-to-date as of at least __function_invoked_time__ (memory-deduced-qualified (void))
    inline void virtual_page_drop(size_t page_idx) noexcept{

        while (!virtual_page_try_release_if_zero_ref(page_idx)){} //schedulers here 
    }

    //wait + sync the page_idx virtual_page. the exit of this function guarantees the up-to-date as of at least __function_invoked_time__ (memory-deduced-qualified (void))
    inline void virtual_page_sync(size_t page_idx) noexcept{

        while (!virtual_page_try_sync(page_idx)){} //schedulers here
    }

    //wait + drop all pages. the exit of this function guarantees the up-to-date as of at least __function_invoked_time__ (memory-deduced-qualified (void))
    inline void virtual_page_drop_all() noexcept{

        for (size_t i = 0; i < table.virtual_page_list_sz; ++i){
            virtual_page_drop(i);
        }
    }

    //wait + sync all pages. the exit of this function guarantees the up-to-date as of at least __function_invoked_time__ ...
    inline void virtual_page_sync_all() noexcept{

        for (size_t i = 0; i < table.virtual_page_list_sz; ++i){
            virtual_page_sync(i);
        }
    }
    
    //--user-interface--

    //should be invoked once - at the beginning of the program
    inline void init(char * translator_addr, size_t translator_sz,
                     char * translatee_addr, size_t translatee_sz,
                     mem_transfer_device_t virtual_to_physical_transfer_device,
                     mem_transfer_device_t physical_to_virtual_transfer_device){
                    
        if (translator_sz % PAGE_SZ != 0u || translator_sz == 0u || reinterpret_cast<uintptr_t>(translator_addr) % PAGE_SZ != 0u || reinterpret_cast<uintptr_t>(translator_addr) / PAGE_SZ == 0u){ //page_offs != 0, bad practice - but necessary for remapping nullptr
            std::abort();
        }

        if (translatee_sz % PAGE_SZ != 0u || translatee_sz == 0u || reinterpret_cast<uintptr_t>(translatee_addr) % PAGE_SZ != 0u || reinterpret_cast<uintptr_t>(translatee_addr) / PAGE_SZ == 0u){ //page_offs != 0, bad practice - but necessary for remapping nullptr
            std::abort();
        }

        size_t translator_page_count    = translator_sz / PAGE_SZ;
        auto translator_pages           = std::make_unique<VirtualPageState[]>(translator_page_count);
        size_t translatee_page_count    = translatee_sz / PAGE_SZ;
        auto translatee_pages           = std::make_unique<PhysicalPageState[]>(translatee_page_count);

        for (size_t i = 0; i < translator_page_count; ++i){
            dg_atomic_exchange(translator_pages[i].state, virtual_page_null_state, std::memory_order_seq_cst);
        }

        for (size_t i = 0; i < translatee_page_count; ++i){
            translatee_pages[i].addr = translatee_addr + (PAGE_SZ * i); 
            dg_atomic_flag_clear(translatee_pages[i].is_acquired, std::memory_order_seq_cst);
        }

        config                      = {translator_addr, translator_sz, translatee_addr, translatee_sz, virtual_to_physical_transfer_device, physical_to_virtual_transfer_device};
        table.virtual_page_list_sz  = translator_page_count;
        table.virtual_page_list     = translator_pages.get();
        table.physical_page_list_sz = translatee_page_count;
        table.physical_page_list    = translatee_pages.get();

        translator_pages.release();
        translatee_pages.release();
    }

    inline auto map(void * ptr) -> void *{

        if (!ptr){
            return nullptr;
        }

        size_t idx              = std::distance(static_cast<const char *>(config.translator_addr), static_cast<const char *>(ptr));
        size_t page_slot        = slot(idx, PAGE_SZ);
        size_t page_offs        = offset(idx, PAGE_SZ);
        void * translatee_page  = virtual_page_force_fetch_n_inc_ref(page_slot);  

        return static_cast<char *>(translatee_page) + page_offs;
    } 

    inline void unmap(void * ptr) noexcept{

        if (!ptr){
            return;
        }

        size_t idx          = std::distance(static_cast<const char *>(config.translator_addr), static_cast<const char *>(ptr));
        size_t page_slot    = slot(idx, PAGE_SZ);
        virtual_page_dec_ref(page_slot);
    }

    inline void shootdown(void * ptr) noexcept{

        if (!ptr){
            return;
        }

        size_t idx          = std::distance(static_cast<const char *>(config.translator_addr), static_cast<const char *>(ptr));
        size_t page_slot    = slot(idx, PAGE_SZ);
        virtual_page_drop(page_slot);
    }

    inline void sync(void * ptr) noexcept{

        if (!ptr){
            return; 
        }

        size_t idx          = std::distance(static_cast<const char *>(config.translator_addr), static_cast<const char *>(ptr));
        size_t page_slot    = slot(idx, PAGE_SZ);
        virtual_page_sync(page_slot);
    }

    inline void flush() noexcept{

        virtual_page_drop_all();
    }

    inline void sync() noexcept{

        virtual_page_sync_all();
    }

    inline auto remap(void * old_ptr, void * old_mapped_ptr, void * new_ptr) -> void *{

        //consider branchless - should be compiler's optimization work in the future(if not already now) 
        if (slot(reinterpret_cast<uintptr_t>(old_ptr), PAGE_SZ) == slot(reinterpret_cast<uintptr_t>(new_ptr), PAGE_SZ)){ 
            return static_cast<char *>(old_mapped_ptr) + std::distance(static_cast<const char *>(old_ptr), static_cast<const char *>(new_ptr)); //UB
        }

        void * rs = map(new_ptr);
        unmap(old_ptr);
        return rs;
    }

}

#endif