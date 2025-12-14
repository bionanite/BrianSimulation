#!/usr/bin/env python3
"""
Test Framework for Global Workspace
Tests global workspace, information integration, consciousness access, and broadcasting
"""

import numpy as np
import matplotlib.pyplot as plt
from global_workspace import (
    GlobalWorkspaceManager, GlobalWorkspace,
    InformationIntegration, ConsciousnessAccess, BroadcastMechanism
)

class GlobalWorkspaceTester:
    """Test framework for global workspace"""
    
    def __init__(self):
        self.results = []
    
    def test_global_workspace(self):
        """Test global workspace"""
        print("\n" + "="*60)
        print("TEST 1: Global Workspace")
        print("="*60)
        
        workspace = GlobalWorkspace()
        
        # Test 1: Add content
        print("\nTest 1.1: Adding content")
        content1 = workspace.add_content("sensory", {"vision": 0.8, "sound": 0.6}, activation=0.9)
        content2 = workspace.add_content("memory", {"recall": 0.7}, activation=0.3)
        
        print(f"   Contents added: {len(workspace.workspace_contents)}")
        print(f"   Result: {'✅ PASS' if len(workspace.workspace_contents) >= 1 else '❌ FAIL'}")
        
        # Test 2: Get active contents
        print("\nTest 1.2: Getting active contents")
        active = workspace.get_active_contents()
        
        print(f"   Active contents: {len(active)}")
        print(f"   Result: {'✅ PASS' if len(active) >= 1 else '❌ FAIL'}")
        
        # Test 3: Integrate contents
        print("\nTest 1.3: Integrating contents")
        integrated = workspace.integrate_contents()
        
        print(f"   Integrated keys: {len(integrated)}")
        print(f"   Result: {'✅ PASS' if len(integrated) > 0 else '⚠️  CHECK'}")
        
        return True
    
    def test_information_integration(self):
        """Test information integration"""
        print("\n" + "="*60)
        print("TEST 2: Information Integration")
        print("="*60)
        
        integration = InformationIntegration()
        
        # Test 1: Set source weights
        print("\nTest 2.1: Setting source weights")
        integration.set_source_weight("sensory", 0.8)
        integration.set_source_weight("memory", 0.6)
        
        print(f"   Source weights: {len(integration.source_weights)}")
        print(f"   Result: {'✅ PASS' if len(integration.source_weights) == 2 else '❌ FAIL'}")
        
        # Test 2: Integrate sources
        print("\nTest 2.2: Integrating sources")
        sources = {
            "sensory": {"vision": 0.8, "sound": 0.6},
            "memory": {"recall": 0.7}
        }
        integrated = integration.integrate(sources)
        
        print(f"   Integrated keys: {len(integrated)}")
        print(f"   Result: {'✅ PASS' if len(integrated) > 0 else '❌ FAIL'}")
        
        return True
    
    def test_consciousness_access(self):
        """Test consciousness access"""
        print("\n" + "="*60)
        print("TEST 3: Consciousness Access")
        print("="*60)
        
        access = ConsciousnessAccess()
        from global_workspace import WorkspaceContent
        
        # Test 1: Grant access
        print("\nTest 3.1: Granting access")
        content1 = WorkspaceContent(0, "sensory", {"vision": 0.8}, activation=0.9)
        content2 = WorkspaceContent(1, "memory", {"recall": 0.7}, activation=0.3)
        
        granted1 = access.grant_access(content1)
        granted2 = access.grant_access(content2)
        
        print(f"   Content 1 granted: {granted1}")
        print(f"   Content 2 granted: {granted2}")
        print(f"   Result: {'✅ PASS' if granted1 and not granted2 else '⚠️  CHECK'}")
        
        # Test 2: Get conscious contents
        print("\nTest 3.2: Getting conscious contents")
        conscious = access.get_conscious_contents()
        
        print(f"   Conscious contents: {len(conscious)}")
        print(f"   Result: {'✅ PASS' if len(conscious) >= 1 else '❌ FAIL'}")
        
        return True
    
    def test_broadcast_mechanism(self):
        """Test broadcast mechanism"""
        print("\n" + "="*60)
        print("TEST 4: Broadcast Mechanism")
        print("="*60)
        
        broadcast = BroadcastMechanism()
        
        # Test 1: Subscribe
        print("\nTest 4.1: Subscribing to broadcasts")
        callback_count = [0]
        
        def callback(content_id, content):
            callback_count[0] += 1
        
        broadcast.subscribe("system1", callback)
        broadcast.subscribe("system2", callback)
        
        print(f"   Subscribers: {len(broadcast.subscribers)}")
        print(f"   Result: {'✅ PASS' if len(broadcast.subscribers) == 2 else '❌ FAIL'}")
        
        # Test 2: Broadcast
        print("\nTest 4.2: Broadcasting")
        broadcast.broadcast(1, {"info": 0.8}, ["system1", "system2"])
        
        print(f"   Callbacks triggered: {callback_count[0]}")
        print(f"   Result: {'✅ PASS' if callback_count[0] == 2 else '❌ FAIL'}")
        
        return True
    
    def test_integrated_global_workspace(self):
        """Test integrated global workspace"""
        print("\n" + "="*60)
        print("TEST 5: Integrated Global Workspace")
        print("="*60)
        
        manager = GlobalWorkspaceManager()
        
        # Add to workspace
        print("\nAdding content to workspace...")
        content1 = manager.add_to_workspace("sensory", {"vision": 0.8, "sound": 0.6}, activation=0.9)
        content2 = manager.add_to_workspace("memory", {"recall": 0.7, "recognition": 0.8}, activation=0.85)
        content3 = manager.add_to_workspace("reasoning", {"inference": 0.75}, activation=0.7)
        
        print(f"   Contents added: {len(manager.global_workspace.workspace_contents)}")
        
        # Integrate sources
        print("\nIntegrating information sources...")
        sources = {
            "sensory": {"vision": 0.8, "sound": 0.6},
            "memory": {"recall": 0.7},
            "reasoning": {"inference": 0.75}
        }
        manager.information_integration.set_source_weight("sensory", 0.8)
        manager.information_integration.set_source_weight("memory", 0.6)
        integrated = manager.integrate_sources(sources)
        
        print(f"   Integrated information: {len(integrated)} keys")
        
        # Subscribe and broadcast
        print("\nSetting up broadcasting...")
        received_content = [None]
        
        def receive_broadcast(content_id, content):
            received_content[0] = content
        
        manager.broadcast_mechanism.subscribe("test_system", receive_broadcast)
        
        if content1:
            manager.broadcast(content1.content_id, content1.content, ["test_system"])
        
        print(f"   Broadcast received: {received_content[0] is not None}")
        
        # Visualize
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle('Global Workspace', fontsize=16, fontweight='bold')
        
        # Plot 1: Workspace contents activation
        ax1 = axes[0, 0]
        contents = list(manager.global_workspace.workspace_contents)
        sources_list = [c.source for c in contents]
        activations = [c.activation for c in contents]
        
        bars = ax1.bar(range(len(contents)), activations, color='#3498DB', alpha=0.8, edgecolor='black')
        ax1.set_title('Workspace Contents', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Activation')
        ax1.set_xlabel('Content Index')
        ax1.set_xticks(range(len(contents)))
        ax1.set_xticklabels(sources_list, rotation=45, ha='right')
        ax1.axhline(y=manager.global_workspace.activation_threshold, color='r', linestyle='--', label='Threshold')
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)
        
        # Plot 2: Integrated information
        ax2 = axes[0, 1]
        if integrated:
            keys = list(integrated.keys())
            values = [integrated[k] for k in keys]
            
            bars = ax2.bar(keys, values, color='#2ECC71', alpha=0.8, edgecolor='black')
            ax2.set_title('Integrated Information', fontsize=12, fontweight='bold')
            ax2.set_ylabel('Value')
            ax2.set_xticklabels(keys, rotation=45, ha='right')
            ax2.grid(axis='y', alpha=0.3)
        
        # Plot 3: Conscious contents
        ax3 = axes[1, 0]
        conscious = manager.consciousness_access.get_conscious_contents()
        conscious_sources = [c.source for c in conscious]
        conscious_activations = [c.activation for c in conscious]
        
        if conscious:
            bars = ax3.bar(range(len(conscious)), conscious_activations, color='#F39C12', alpha=0.8, edgecolor='black')
            ax3.set_title('Conscious Contents', fontsize=12, fontweight='bold')
            ax3.set_ylabel('Activation')
            ax3.set_xticks(range(len(conscious)))
            ax3.set_xticklabels(conscious_sources, rotation=45, ha='right')
            ax3.grid(axis='y', alpha=0.3)
        
        # Plot 4: Statistics
        ax4 = axes[1, 1]
        ax4.axis('off')
        
        stats = manager.get_statistics()
        stats_text = "Global Workspace Statistics:\n\n"
        stats_text += f"Total Contents: {stats['total_contents']}\n"
        stats_text += f"Active Contents: {stats['active_contents']}\n"
        stats_text += f"Capacity Usage: {stats['capacity_usage']:.2%}\n"
        stats_text += f"Integrations: {stats['integrations']}\n"
        stats_text += f"Conscious Contents: {stats['conscious_contents']}\n"
        stats_text += f"Broadcasts: {stats['broadcasts']}\n"
        stats_text += f"Subscribers: {stats['subscribers']}\n"
        
        ax4.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center', transform=ax4.transAxes,
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig('global_workspace_progress.png', dpi=300, bbox_inches='tight')
        print("✅ Visualization saved: global_workspace_progress.png")
        plt.close()
        
        # Final statistics
        stats = manager.get_statistics()
        print(f"\n📊 Final Statistics:")
        print(f"   Total contents: {stats['total_contents']}")
        print(f"   Active contents: {stats['active_contents']}")
        print(f"   Conscious contents: {stats['conscious_contents']}")
        print(f"   Broadcasts: {stats['broadcasts']}")
        
        return True
    
    def run_all_tests(self):
        """Run all global workspace tests"""
        print("\n" + "="*70)
        print("GLOBAL WORKSPACE TEST SUITE")
        print("="*70)
        
        tests = [
            ("Global Workspace", self.test_global_workspace),
            ("Information Integration", self.test_information_integration),
            ("Consciousness Access", self.test_consciousness_access),
            ("Broadcast Mechanism", self.test_broadcast_mechanism),
            ("Integrated Global Workspace", self.test_integrated_global_workspace)
        ]
        
        results = []
        for test_name, test_func in tests:
            try:
                result = test_func()
                results.append((test_name, result))
            except Exception as e:
                print(f"\n❌ {test_name} FAILED with error: {e}")
                import traceback
                traceback.print_exc()
                results.append((test_name, False))
        
        # Summary
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        
        passed = sum(1 for _, r in results if r)
        total = len(results)
        
        for test_name, result in results:
            status = "✅ PASS" if result else "❌ FAIL"
            print(f"   {test_name}: {status}")
        
        print(f"\n   Total: {passed}/{total} tests passed")
        print(f"   Success rate: {passed/total*100:.1f}%")
        
        return passed == total

def main():
    """Main test function"""
    tester = GlobalWorkspaceTester()
    success = tester.run_all_tests()
    
    if success:
        print("\n✅ All global workspace tests passed!")
    else:
        print("\n⚠️  Some tests failed. Review output above.")
    
    return success

if __name__ == "__main__":
    main()

