#include "NativeCachePolicy.hpp"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#include <os/proc.h>
#include <TargetConditionals.h>
#include <mach/mach.h>

namespace iiLocalDiffusion::native_detail {
ResourceLimits resourceLimits() {
    @autoreleasepool {
        ResourceLimits limits;
        limits.physical = NSProcessInfo.processInfo.physicalMemory;
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        limits.recommended = device.recommendedMaxWorkingSetSize;
#if TARGET_OS_IOS
        limits.available = os_proc_available_memory();
        limits.availableKnown = true;
#else
        // Metal's recommendation is not currently free system RAM. Respect
        // other apps on macOS instead of forcing model residency into swap.
        const auto host = mach_host_self();
        vm_size_t pageSize = 0;
        vm_statistics64_data_t stats{};
        mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
        if (host_page_size(host, &pageSize) == KERN_SUCCESS &&
            host_statistics64(host, HOST_VM_INFO64, reinterpret_cast<host_info64_t>(&stats), &count) == KERN_SUCCESS) {
            limits.available = (static_cast<std::uint64_t>(stats.free_count) + stats.inactive_count) * pageSize;
            limits.availableKnown = true;
        }
        mach_port_deallocate(mach_task_self(), host);
#endif
        return limits;
    }
}
}
