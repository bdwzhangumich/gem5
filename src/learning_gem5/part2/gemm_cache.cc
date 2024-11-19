#include "learning_gem5/part2/gemm_cache.hh"

#include "base/compiler.hh"
#include "base/random.hh"
#include "debug/GeMMCache.hh"
#include "sim/system.hh"

namespace gem5
{

GeMMCache::GeMMCache(const GeMMCacheParams &params) :
    ClockedObject(params),
    latency(params.latency),
	cacheAddr(params.system->cacheStartingAddress()),
    matrixBytes(params.system->matrixSizeBytes()), // assume a power of 2
	numMatrices(params.system->numberMatrices()), // assume a power of 2
    memPort(params.name + ".mem_side", this),
    blocked(false), originalPacket(nullptr), waitingPortId(-1), stats(this)
{
    // Since the CPU side ports are a vector of ports, create an instance of
    // the CPUSidePort for each connection. This member of params is
    // automatically created depending on the name of the vector port and
    // holds the number of connections to this port name
    for (int i = 0; i < params.port_cpu_side_connection_count; ++i) {
        cpuPorts.emplace_back(name() + csprintf(".cpu_side[%d]", i), i, this);
    }
}

Port &
GeMMCache::getPort(const std::string &if_name, PortID idx)
{
    // This is the name from the Python SimObject declaration in SimpleCache.py
    if (if_name == "mem_side") {
        panic_if(idx != InvalidPortID,
                 "Mem side of simple cache not a vector port");
        return memPort;
    } else if (if_name == "cpu_side" && idx < cpuPorts.size()) {
        // We should have already created all of the ports in the constructor
        return cpuPorts[idx];
    } else {
        // pass it along to our super class
        return ClockedObject::getPort(if_name, idx);
    }
}

void
GeMMCache::CPUSidePort::sendPacket(PacketPtr pkt)
{
    // Note: This flow control is very simple since the cache is blocking.

    panic_if(blockedPacket != nullptr, "Should never try to send if blocked!");

    // If we can't send the packet across the port, store it for later.
    DPRINTF(GeMMCache, "Sending %s to CPU\n", pkt->print());
    if (!sendTimingResp(pkt)) {
        DPRINTF(GeMMCache, "failed!\n");
        blockedPacket = pkt;
    }
}

AddrRangeList
GeMMCache::CPUSidePort::getAddrRanges() const
{
    return owner->getAddrRanges();
}

void
GeMMCache::CPUSidePort::trySendRetry()
{
    if (needRetry && blockedPacket == nullptr) {
        // Only send a retry if the port is now completely free
        needRetry = false;
        DPRINTF(GeMMCache, "Sending retry req.\n");
        sendRetryReq();
    }
}

void
GeMMCache::CPUSidePort::recvFunctional(PacketPtr pkt)
{
    // Just forward to the cache.
    return owner->handleFunctional(pkt);
}

bool
GeMMCache::CPUSidePort::recvTimingReq(PacketPtr pkt)
{
    DPRINTF(GeMMCache, "Got request %s\n", pkt->print());

    if (blockedPacket || needRetry) {
        // The cache may not be able to send a reply if this is blocked
        DPRINTF(GeMMCache, "Request blocked\n");
        needRetry = true;
        return false;
    }
    
	bool handled = false;
    // We need address to determine where to send data
	if (pkt->getAddr() > cacheAddr + (numMatrices * matrixBytes) || pkt->getAddr() < cacheAddr)
        handled = owner->handleMemoryRequest(pkt, id); // Forward to memory
	else
		handled = owner->handleCacheRequest(pkt, id); // Handle in cache.
    
	if (!handled) {
		DPRINTF(GeMMCache, "Request failed\n");
		// stalling
		needRetry = true;
		return false;
	} else {
		DPRINTF(GeMMCache, "Request succeeded\n");
		return true;
	}
}

void
GeMMCache::CPUSidePort::recvRespRetry()
{
    // We should have a blocked packet if this function is called.
    assert(blockedPacket != nullptr);

    // Grab the blocked packet.
    PacketPtr pkt = blockedPacket;
    blockedPacket = nullptr;

    DPRINTF(GeMMCache, "Retrying response pkt %s\n", pkt->print());
    // Try to resend it. It's possible that it fails again.
    sendPacket(pkt);

    // We may now be able to accept new packets
    trySendRetry();
}

void
GeMMCache::MemSidePort::sendPacket(PacketPtr pkt)
{
    // Note: This flow control is very simple since the cache is blocking.

    panic_if(blockedPacket != nullptr, "Should never try to send if blocked!");

    // If we can't send the packet across the port, store it for later.
    if (!sendTimingReq(pkt)) {
        blockedPacket = pkt;
    }
}

bool
GeMMCache::MemSidePort::recvTimingResp(PacketPtr pkt)
{
	// TODO: figure out what this function should do, probably send back to CPU
    // Just forward to the cache.
    return owner->handleResponse(pkt);
}

void
GeMMCache::MemSidePort::recvReqRetry()
{
    // We should have a blocked packet if this function is called.
    assert(blockedPacket != nullptr);

    // Grab the blocked packet.
    PacketPtr pkt = blockedPacket;
    blockedPacket = nullptr;

    // Try to resend it. It's possible that it fails again.
    sendPacket(pkt);
}

void
GeMMCache::MemSidePort::recvRangeChange()
{
    owner->sendRangeChange();
}

bool
GeMMCache::MemSidePort::handleMemoryRequest(PacketPtr pkt, int port_id)
{
	if (blocked) {
        // There is currently an outstanding request. Stall.
        return false;
    }

    DPRINTF(GeMMCache, "Got request for addr %#x\n", pkt->getAddr());

    // This memobj is now blocked waiting for the response to this packet.
    blocked = true;

    // Simply forward to the memory port
    sendPacket(pkt);

    return true;
}

bool
GeMMCache::handleCacheRequest(PacketPtr pkt, int port_id)
{
	// TODO: implement handleCacheRequest for memory mapped cache 
    if (blocked) {
        // There is currently an outstanding request so we can't respond. Stall
        return false;
    }

    DPRINTF(GeMMCache, "Got request for addr %#x\n", pkt->getAddr());

    // This cache is now blocked waiting for the response to this packet.
    blocked = true;

    // Store the port for when we get the response
    assert(waitingPortId == -1);
    waitingPortId = port_id;

    // Schedule an event after cache access latency to actually access
    schedule(new EventFunctionWrapper([this, pkt]{ accessTiming(pkt); },
                                      name() + ".accessEvent", true),
             clockEdge(latency));

    return true;
}

bool
GeMMCache::handleResponse(PacketPtr pkt)
{
    assert(blocked);
    DPRINTF(GeMMCache, "Got response for addr %#x\n", pkt->getAddr());

    // For now assume that inserts are off of the critical path and don't count
    // for any added latency.
    insert(pkt);

    stats.missLatency.sample(curTick() - missTime);

    // If we had to upgrade the request packet to a full cache line, now we
    // can use that packet to construct the response.
    if (originalPacket != nullptr) {
        DPRINTF(GeMMCache, "Copying data from new packet to old\n");
        // We had to upgrade a previous packet. We can functionally deal with
        // the cache access now. It better be a hit.
        [[maybe_unused]] bool hit = accessFunctional(originalPacket);
        panic_if(!hit, "Should always hit after inserting");
        originalPacket->makeResponse();
        delete pkt; // We may need to delay this, I'm not sure.
        pkt = originalPacket;
        originalPacket = nullptr;
    } // else, pkt contains the data it needs

    sendResponse(pkt);

    return true;
}

void GeMMCache::sendResponse(PacketPtr pkt)
{
    assert(blocked);
    DPRINTF(GeMMCache, "Sending resp for addr %#x\n", pkt->getAddr());

    int port = waitingPortId;

    // The packet is now done. We're about to put it in the port, no need for
    // this object to continue to stall.
    // We need to free the resource before sending the packet in case the CPU
    // tries to send another request immediately (e.g., in the same callchain).
    blocked = false;
    waitingPortId = -1;

    // Simply forward to the memory port
    cpuPorts[port].sendPacket(pkt);

    // For each of the cpu ports, if it needs to send a retry, it should do it
    // now since this memory object may be unblocked now.
    for (auto& port : cpuPorts) {
        port.trySendRetry();
    }
}

void
GeMMCache::handleFunctional(PacketPtr pkt)
{
    if (accessFunctional(pkt)) {
        pkt->makeResponse();
    } else {
        memPort.sendFunctional(pkt);
    }
}

void
GeMMCache::accessTiming(PacketPtr pkt)
{
    bool hit = accessFunctional(pkt);

    DPRINTF(GeMMCache, "%s for packet: %s\n", hit ? "Hit" : "Miss",
            pkt->print());

    if (hit) {
        // Respond to the CPU side
        stats.hits++; // update stats
        DDUMP(GeMMCache, pkt->getConstPtr<uint8_t>(), pkt->getSize());
        pkt->makeResponse();
        sendResponse(pkt);
    } else {
        stats.misses++; // update stats
        missTime = curTick();
        // Forward to the memory side.
        // We can't directly forward the packet unless it is exactly the size
        // of the cache line, and aligned. Check for that here.
        Addr addr = pkt->getAddr();
        Addr block_addr = pkt->getBlockAddr(blockSize);
        unsigned size = pkt->getSize();
        if (addr == block_addr && size == blockSize) {
            // Aligned and block size. We can just forward.
            DPRINTF(GeMMCache, "forwarding packet\n");
            memPort.sendPacket(pkt);
        } else {
            DPRINTF(GeMMCache, "Upgrading packet to block size\n");
            panic_if(addr - block_addr + size > blockSize,
                     "Cannot handle accesses that span multiple cache lines");
            // Unaligned access to one cache block
            assert(pkt->needsResponse());
            MemCmd cmd;
            if (pkt->isWrite() || pkt->isRead()) {
                // Read the data from memory to write into the block.
                // We'll write the data in the cache (i.e., a writeback cache)
                cmd = MemCmd::ReadReq;
            } else {
                panic("Unknown packet type in upgrade size");
            }

            // Create a new packet that is blockSize
            PacketPtr new_pkt = new Packet(pkt->req, cmd, blockSize);
            new_pkt->allocate();

            // Should now be block aligned
            assert(new_pkt->getAddr() == new_pkt->getBlockAddr(blockSize));

            // Save the old packet
            originalPacket = pkt;

            DPRINTF(GeMMCache, "forwarding packet\n");
            memPort.sendPacket(new_pkt);
        }
    }
}

bool
GeMMCache::accessFunctional(PacketPtr pkt)
{
    Addr block_addr = pkt->getBlockAddr(blockSize);
    auto it = cacheStore.find(block_addr);
    if (it != cacheStore.end()) {
        if (pkt->isWrite()) {
            // Write the data into the block in the cache
            pkt->writeDataToBlock(it->second, blockSize);
        } else if (pkt->isRead()) {
            // Read the data out of the cache block into the packet
            pkt->setDataFromBlock(it->second, blockSize);
        } else {
            panic("Unknown packet type!");
        }
        return true;
    }
    return false;
}

void
GeMMCache::insert(PacketPtr pkt)
{
    // The packet should be aligned.
    assert(pkt->getAddr() ==  pkt->getBlockAddr(blockSize));
    // The address should not be in the cache
    assert(cacheStore.find(pkt->getAddr()) == cacheStore.end());
    // The pkt should be a response
    assert(pkt->isResponse());

    if (cacheStore.size() >= capacity) {
        // Select random thing to evict. This is a little convoluted since we
        // are using a std::unordered_map. See http://bit.ly/2hrnLP2
        int bucket, bucket_size;
        do {
            bucket = random_mt.random(0, (int)cacheStore.bucket_count() - 1);
        } while ( (bucket_size = cacheStore.bucket_size(bucket)) == 0 );
        auto block = std::next(cacheStore.begin(bucket),
                               random_mt.random(0, bucket_size - 1));

        DPRINTF(GeMMCache, "Removing addr %#x\n", block->first);

        // Write back the data.
        // Create a new request-packet pair
        RequestPtr req = std::make_shared<Request>(
            block->first, blockSize, 0, 0);

        PacketPtr new_pkt = new Packet(req, MemCmd::WritebackDirty, blockSize);
        new_pkt->dataDynamic(block->second); // This will be deleted later

        DPRINTF(GeMMCache, "Writing packet back %s\n", pkt->print());
        // Send the write to memory
        memPort.sendPacket(new_pkt);

        // Delete this entry
        cacheStore.erase(block->first);
    }

    DPRINTF(GeMMCache, "Inserting %s\n", pkt->print());
    DDUMP(GeMMCache, pkt->getConstPtr<uint8_t>(), blockSize);

    // Allocate space for the cache block data
    uint8_t *data = new uint8_t[blockSize];

    // Insert the data and address into the cache store
    cacheStore[pkt->getAddr()] = data;

    // Write the data into the cache
    pkt->writeDataToBlock(data, blockSize);
}

AddrRangeList
GeMMCache::getAddrRanges() const
{
    DPRINTF(GeMMCache, "Sending new ranges\n");
    // Just use the same ranges as whatever is on the memory side.
    return memPort.getAddrRanges();
}

void
GeMMCache::sendRangeChange() const
{
    for (auto& port : cpuPorts) {
        port.sendRangeChange();
    }
}

GeMMCache::GeMMCacheStats::GeMMCacheStats(statistics::Group *parent)
      : statistics::Group(parent),
      ADD_STAT(hits, statistics::units::Count::get(), "Number of hits"),
      ADD_STAT(misses, statistics::units::Count::get(), "Number of misses"),
      ADD_STAT(missLatency, statistics::units::Tick::get(),
               "Ticks for misses to the cache"),
      ADD_STAT(hitRatio, statistics::units::Ratio::get(),
               "The ratio of hits to the total accesses to the cache",
               hits / (hits + misses))
{
    missLatency.init(16); // number of buckets
}

bool
GeMMCache::start_matrix_mult(PacketPtr pkt)
{
    // Addr block_addr = pkt->getBlockAddr(blockSize);
    // auto it = cacheStore.find(block_addr);
    // if (it != cacheStore.end()) {
    //     if (pkt->isWrite()) {
    //         // Write the data into the block in the cache
    //         pkt->writeDataToBlock(it->second, blockSize);
    //     } else if (pkt->isRead()) {
    //         // Read the data out of the cache block into the packet
    //         pkt->setDataFromBlock(it->second, blockSize);
    //     } else {
    //         panic("Unknown packet type!");
    //     }
    //     return true;
    // }
    // return false;
}

bool
GeMMCache::route_packet(PacketPtr pkt)
{
	// Get the address from the packet
	// Send to handle_memory_packet (forward to memory) if outside cache address range, else send to handle_cache_packet
}
bool
GeMMCache::insert_at_addr(PacketPtr pkt) 
{
    // Get the address from the packet
    // Get the data from the packet
	// insert into cache array
	// memory protocol stuff, eg respond to cpu


    // just call access_functional?
}
bool
GeMMCache::load_from_addr(PacketPtr pkt) 
{
    // Get the address from the packet
	// Get data from cache array and put into response packet
	// memory protocol stuff, eg send packet to cpu

    // also just call access functional?
}
} // namespace gem5
