#!/usr/bin/env python3

from pygltflib import GLTF2
import argparse
import json

def main():
    parser = argparse.ArgumentParser(description="Inspect GLB structure")
    parser.add_argument("--input", "-i", required=True, help="Input .glb file")
    args = parser.parse_args()

    gltf = GLTF2().load(args.input)

    print("\n===== SCENES =====")
    for idx, scene in enumerate(gltf.scenes):
        print(f"Scene {idx}: nodes={scene.nodes}")

    print("\n===== NODES =====")
    for idx, node in enumerate(gltf.nodes):
        print(f"Node {idx}:")
        print(f"  name={node.name}")
        print(f"  mesh={node.mesh}")
        print(f"  children={node.children}")
        print(f"  translation={node.translation}")
        print(f"  rotation={node.rotation}")
        print(f"  scale={node.scale}")

    print("\n===== MESHES =====")
    for idx, mesh in enumerate(gltf.meshes):
        print(f"\nMesh {idx}: name={mesh.name}")
        for p_idx, primitive in enumerate(mesh.primitives):
            print(f"  Primitive {p_idx}:")
            print(f"    mode={primitive.mode}")
            print(f"    attributes={primitive.attributes}")
            print(f"    indices={primitive.indices}")
            print(f"    material={primitive.material}")

    print("\n===== ACCESSORS =====")
    for idx, acc in enumerate(gltf.accessors):
        print(f"Accessor {idx}:")
        print(f"  bufferView={acc.bufferView}")
        print(f"  componentType={acc.componentType}")
        print(f"  count={acc.count}")
        print(f"  type={acc.type}")
        print(f"  min={acc.min}")
        print(f"  max={acc.max}")

    print("\n===== BUFFER VIEWS =====")
    for idx, bv in enumerate(gltf.bufferViews):
        print(f"BufferView {idx}: buffer={bv.buffer}, byteOffset={bv.byteOffset}, byteLength={bv.byteLength}, byteStride={bv.byteStride}")

    print("\n===== BUFFERS =====")
    for idx, buf in enumerate(gltf.buffers):
        print(f"Buffer {idx}: byteLength={buf.byteLength}")

    print("\n===== MATERIALS =====")
    for idx, mat in enumerate(gltf.materials):
        print(f"Material {idx}:")
        print(json.dumps(mat.to_dict(), indent=2))

    print("\nDONE.\n")

if __name__ == "__main__":
    main()
