#ifndef __GEMM_CACHE_HH__
#define __GEMM_CACHE_HH__

#include <unordered_map>

#include "base/statistics.hh"
#include "mem/port.hh"
#include "params/GeMMCache.hh"
#include "sim/clocked_object.hh"

namespace gem5
{

/**
 * A very simple cache object. Has a fully-associative data store with random
 * replacement.
 * This cache is fully blocking (not non-blocking). Only a single request can
 * be outstanding at a time.
 * This cache is a writeback cache.
 */
class GeMMCache : public ClockedObject
{
  private:

    /**
     * Port on the CPU-side that receives requests.
     * Mostly just forwards requests to the cache (owner)
     */
    class CPUSidePort : public ResponsePort
    {
      private:
        /// Since this is a vector port, need to know what number this one is
        int id;

        /// The object that owns this object (SimpleCache)
        GeMMCache *owner;

        /// True if the port needs to send a retry req.
        bool needRetry;

        /// If we tried to send a packet and it was blocked, store it here
        PacketPtr blockedPacket;

      public:
        /**
         * Constructor. Just calls the superclass constructor.
         */
        CPUSidePort(const std::string& name, int id, GeMMCache *owner) :
            ResponsePort(name), id(id), owner(owner), needRetry(false),
            blockedPacket(nullptr)
        { }

        /**
         * Send a packet across this port. This is called by the owner and
         * all of the flow control is hanled in this function.
         * This is a convenience function for the SimpleCache to send pkts.
         *
         * @param packet to send.
         */
        void sendPacket(PacketPtr pkt);

        /**
         * Get a list of the non-overlapping address ranges the owner is
         * responsible for. All response ports must override this function
         * and return a populated list with at least one item.
         *
         * @return a list of ranges responded to
         */
        AddrRangeList getAddrRanges() const override;

        /**
         * Send a retry to the peer port only if it is needed. This is called
         * from the SimpleCache whenever it is unblocked.
         */
        void trySendRetry();

      protected:
        /**
         * Receive an atomic request packet from the request port.
         * No need to implement in this simple cache.
         */
        Tick recvAtomic(PacketPtr pkt) override
        { panic("recvAtomic unimpl."); }

        /**
         * Receive a functional request packet from the request port.
         * Performs a "debug" access updating/reading the data in place.
         *
         * @param packet the requestor sent.
         */
        void recvFunctional(PacketPtr pkt) override;

        /**
         * Receive a timing request from the request port.
         *
         * @param the packet that the requestor sent
         * @return whether this object can consume to packet. If false, we
         *         will call sendRetry() when we can try to receive this
         *         request again.
         */
        bool recvTimingReq(PacketPtr pkt) override;

        /**
         * Called by the request port if sendTimingResp was called on this
         * response port (causing recvTimingResp to be called on the request
         * port) and was unsuccessful.
         */
        void recvRespRetry() override;
    };

    /**
     * Port on the memory-side that receives responses.
     * Mostly just forwards requests to the cache (owner)
     */
    class MemSidePort : public RequestPort
    {
      private:
        /// The object that owns this object (SimpleCache)
        GeMMCache *owner;

        /// If we tried to send a packet and it was blocked, store it here
        PacketPtr blockedPacket;

      public:
        /**
         * Constructor. Just calls the superclass constructor.
         */
        MemSidePort(const std::string& name, GeMMCache *owner) :
            RequestPort(name), owner(owner), blockedPacket(nullptr)
        { }

        /**
         * Send a packet across this port. This is called by the owner and
         * all of the flow control is hanled in this function.
         * This is a convenience function for the SimpleCache to send pkts.
         *
         * @param packet to send.
         */
        void sendPacket(PacketPtr pkt);

      protected:
        /**
         * Receive a timing response from the response port.
         */
        bool recvTimingResp(PacketPtr pkt) override;

        /**
         * Called by the response port if sendTimingReq was called on this
         * request port (causing recvTimingReq to be called on the response
         * port) and was unsuccesful.
         */
        void recvReqRetry() override;

        /**
         * Called to receive an address range change from the peer response
         * port. The default implementation ignores the change and does
         * nothing. Override this function in a derived class if the owner
         * needs to be aware of the address ranges, e.g. in an
         * interconnect component like a bus.
         */
        void recvRangeChange() override;
    };

    /**
     * Handle the request to the cache from the CPU side. Called from the CPU port
     * on a timing request.
     *
     * @param requesting packet
     * @param id of the port to send the response
     * @return true if we can handle the request this cycle, false if the
     *         requestor needs to retry later
     */
    bool handleCacheRequest(PacketPtr pkt, int port_id);

    /**
     * Handle the request to DRAM from the CPU side. Called from the CPU port
     * on a timing request.
     *
     * @param requesting packet
     * @param id of the port to send the response
     * @return true if we can handle the request this cycle, false if the
     *         requestor needs to retry later
     */
    bool handleMemoryRequest(PacketPtr pkt, int port_id);

    /**
     * Handle the respone from the memory side. Called from the memory port
     * on a timing response.
     *
     * @param responding packet
     * @return true if we can handle the response this cycle, false if the
     *         responder needs to retry later
     */
    bool handleResponse(PacketPtr pkt);

    /**
     * Send the packet to the CPU side.
     * This function assumes the pkt is already a response packet and forwards
     * it to the correct port. This function also unblocks this object and
     * cleans up the whole request.
     *
     * @param the packet to send to the cpu side
     */
    void sendResponse(PacketPtr pkt);

    /**
     * Handle a packet functionally. Update the data on a write and get the
     * data on a read. Called from CPU port on a recv functional.
     *
     * @param packet to functionally handle
     */
    void handleFunctional(PacketPtr pkt);

    /**
     * Access the cache for a timing access. This is called after the cache
     * access latency has already elapsed.
     */
    void accessTiming(PacketPtr pkt);

    /**
     * This is where we actually update / read from the cache. This function
     * is executed on both timing and functional accesses.
     *
     * @return true if a hit, false otherwise
     */
    bool accessFunctional(PacketPtr pkt);

    /**
     * Insert a block into the cache. If there is no room left in the cache,
     * then this function evicts a random entry t make room for the new block.
     *
     * @param packet with the data (and address) to insert into the cache
     */
    void insert(PacketPtr pkt);

    /**
     * Return the address ranges this cache is responsible for. Just use the
     * same as the next upper level of the hierarchy.
     *
     * @return the address ranges this cache is responsible for
     */
    AddrRangeList getAddrRanges() const;

    /**
     * Tell the CPU side to ask for our memory ranges.
     */
    void sendRangeChange() const;

    /// Latency to check the cache. Number of cycles for both hit and miss
    const Cycles latency;

    /// Instantiation of the CPU-side port
    std::vector<CPUSidePort> cpuPorts;

    /// Instantiation of the memory-side port
    MemSidePort memPort;

    

    /// Packet that we are currently handling. Used for upgrading to larger
    /// cache line sizes
    PacketPtr originalPacket;

    /// The port to send the response when we recieve it back
    int waitingPortId;

    /// For tracking the miss latency
    Tick missTime;

    /// An incredibly simple cache storage. Maps block addresses to data
    std::vector<uint8_t> cacheArray;

    /// Cache statistics
  protected:
    struct GeMMCacheStats : public statistics::Group
    {
        GeMMCacheStats(statistics::Group *parent);
        statistics::Scalar hits;
        statistics::Scalar misses;
        statistics::Histogram missLatency;
        statistics::Formula hitRatio;
    } stats;

    /// The starting address of the cache
    const Addr cacheAddr;

    /// The size of each matrix cache (power of 2)
    const Addr matrixBytes;

    /// The number of matrices in the cache (power of 2)
    const Addr numMatrices;

    /// True if the CPU has an ongoing request waiting for a response.
    bool blocked;

  public:

    /** constructor
     */
    GeMMCache(const GeMMCacheParams &params);

    /**
     * Get a port with a given name and index. This is used at
     * binding time and returns a reference to a protocol-agnostic
     * port.
     *
     * @param if_name Port name
     * @param idx Index in the case of a VectorPort
     *
     * @return A reference to the given port
     */
    Port &getPort(const std::string &if_name,
                  PortID idx=InvalidPortID) override;

};

} // namespace gem5

#endif