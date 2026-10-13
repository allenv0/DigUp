import Testing
@testable import MLXRuntime

/// The target exists; its suites live beside it.
@Suite struct MLXRuntimeTests {
    @Test func targetIsHere() {
        #expect(!MLXModels.cacheName.isEmpty)
    }
}
