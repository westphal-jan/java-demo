package demo;

import org.pytorch.IValue;
import org.pytorch.Module;
import org.pytorch.Tensor;

import java.util.Arrays;

public class App {
  public static void main(String[] args) {
    Tensor data =
            Tensor.fromBlob(
                    new float[] {1, 2, 3, 4, 5}, // data
                    new long[] {1, 5} // shape
            );
    Module mod1 = Module.load("dummy-model.pt");
    IValue result1 = mod1.forward(IValue.from(data));
    Tensor output1 = result1.toTensor();

    System.out.println("Unquantized model:");
    System.out.println("data: " + Arrays.toString(output1.getDataAsFloatArray()));

    Module mod2 = Module.load("dummy-model-quantized.pt");
    IValue result2 = mod2.forward(IValue.from(data));
    Tensor output2 = result2.toTensor();

    System.out.println("Quantized model:");
    System.out.println("data: " + Arrays.toString(output2.getDataAsFloatArray()));

    // Workaround for https://github.com/facebookincubator/fbjni/issues/25
    System.exit(0);
  }
}
