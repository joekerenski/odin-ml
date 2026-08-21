package main

import ml "../ml"

main :: proc() {

    tensor := ml.randn({3,3}, 0.0, 0.5, requires_grad=false)
    tensor2 := ml.randn({3,3}, 0.0, 0.5, requires_grad=false)

    // tensor3 := ml.realize(ml.matmul(tensor, tensor2))
    tensor3 := ml.matmul(tensor, tensor2)
    ml.print_graph(tensor3)

    // ml.println(tensor3)
}
