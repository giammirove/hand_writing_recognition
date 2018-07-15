using System;
using System.IO;
using System.Drawing;
using System.Threading;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace hand_writing_recognition
{
	public class Neuron
	{
		public double learning_rate = 0.1f;
		public double[] input;
		public double[] w;
		public double output;
		public double err;
		public string type_activation;

		public Neuron(int n_input, string _type_activation)
		{
			w = new double[n_input];
			input = new double[n_input];
			type_activation = _type_activation;
		}

		public void init_w(){
			for(int i = 0; i < w.Length; i++)
			{
				w [i] = GetRandomNumber(-0.2f, 0.2f);
			}
		}

		public double GetRandomNumber(double minimum, double maximum)
		{ 
			long tick = DateTime.Now.Ticks;
			Thread.Sleep(1);
			Random random = new Random((int)(tick & 0xffffffffL) | (int) (tick >> 32));
			return random.NextDouble() * (maximum - minimum) + minimum;
		}

		public double activation(double x){
			if (type_activation == "sigmoide") {
				return 1 / (1 + Math.Exp (-x));
			} else if (type_activation == "relu") {
				if (x < 0)
					return 0;
				else
					return x;
			} else {
				// tanh
				return (Math.Exp (x) - Math.Exp (-x)) / (Math.Exp (x) + Math.Exp (-x));
			}
		}

		public double derivative(double x){
			if (type_activation == "sigmoide") {
				return (x) * (1 - (x));
			} else if (type_activation == "relu") {
				if (x < 0)
					return 0;
				else
					return 1;
			} else {
				// tanh
				return 1 - Math.Pow (activation (x), 2);
			}
		}

		public void feed_forward(Layer layer)
		{
			double a = 0;
			for(int i = 0; i < w.Length; i++)
			{
				a += w[i] * input[i];
			}
			a += layer.bias;
			output = activation(a);
		}

		public void update_weights(double error, Layer layer){
			err = error;
			double delta = err * learning_rate * derivative(output);
			for (int i = 0; i < w.Length; i++) {
				w [i] = w [i] + delta * input [i];
			}
			layer.bias = layer.bias + delta;
		}
	}

	public class Layer
	{
		public Neuron[] neuron;
		public double[] _out;
		public double bias;
		public string type_activation;

		public Layer(int n_input, int n_neuron, string _type_activation)
		{
			neuron = new Neuron[n_neuron];
			for(int i = 0; i < n_neuron; i++)
			{
				if(_type_activation == "softmax")
					neuron[i] = new Neuron(n_input, "sigmoide");
				else 
					neuron[i] = new Neuron(n_input, _type_activation);
			}
			type_activation = _type_activation;
		}

		public double[] output(){
			_out = new double[neuron.Length];

			for (int i = 0; i < neuron.Length; i++) {
				neuron [i].feed_forward (this);
				_out [i] = neuron [i].output;
			}

			if (type_activation == "softmax") {
				double[] exp_out = new double[neuron.Length];
				double sum_exp_out = 0;
				for(int i = 0; i < neuron.Length; i++) {
					exp_out [i] = Math.Exp (_out[i]);
					sum_exp_out += exp_out [i];
				}
				for(int i = 0; i < neuron.Length; i++) {
					_out [i] = exp_out[i] / sum_exp_out;
				}
			}
			return _out;
		}
	}

	public static class Extensions
	{
		public static int ReadBigInt32(this BinaryReader br)
		{
			var bytes = br.ReadBytes(sizeof(Int32));
			if (BitConverter.IsLittleEndian) Array.Reverse(bytes);
			return BitConverter.ToInt32(bytes, 0);
		}

		public static void ForEach<T>(this T[,] source, Action<int, int> action)
		{
			for (int w = 0; w < source.GetLength(0); w++)
			{
				for (int h = 0; h < source.GetLength(1); h++)
				{
					action(w, h);
				}
			}
		}
	}

	public static class MnistReader
	{
		private const string TrainImages = "mnist/train-images.idx3-ubyte";
		private const string TrainLabels = "mnist/train-labels.idx1-ubyte";
		private const string TestImages = "mnist/t10k-images.idx3-ubyte";
		private const string TestLabels = "mnist/t10k-labels.idx1-ubyte";

		public static IEnumerable<ImageMnist> ReadTrainingData()
		{
			foreach (var item in Read(TrainImages, TrainLabels))
			{
				yield return item;
			}
		}

		public static IEnumerable<ImageMnist> ReadTestData()
		{
			foreach (var item in Read(TestImages, TestLabels))
			{
				yield return item;
			}
		}

		private static IEnumerable<ImageMnist> Read(string imagesPath, string labelsPath)
		{
			BinaryReader labels = new BinaryReader(new FileStream(labelsPath, FileMode.Open));
			BinaryReader images = new BinaryReader(new FileStream(imagesPath, FileMode.Open));

			int magicNumber = images.ReadBigInt32();
			int numberOfImages = images.ReadBigInt32();
			int width = images.ReadBigInt32();
			int height = images.ReadBigInt32();

			int magicLabel = labels.ReadBigInt32();
			int numberOfLabels = labels.ReadBigInt32();

			for (int i = 0; i < numberOfImages; i++)
			{
				var bytes = images.ReadBytes(width * height);
				double[] newData = new double[width * height];
				for (int j = 0; j < bytes.Length; j++) {
					newData[j] = Convert.ToDouble(bytes[j]);
				}
				var arr = new byte[height * width];
				yield return new ImageMnist()
				{
					Data = bytes,
					Label = labels.ReadByte()
				};
			}

			labels.Close ();
			images.Close ();
		}
	}

	public struct mnist_data
	{
		public string label;
		public double[] rgb;
	}

	public class ImageMnist
	{
		public byte Label { get; set; }
		public byte[] Data { get; set; }
	}

	class Program
	{
		// NON TOCCARE 
		public static int epoche = 1; // 1000
		const int n_neuron_hidden = 20; // 10
		const int n_hidden_layer = 1;
		const int n_x = 784;
		//static int n_training = 3;

		// numero di numeri che sa leggere
		const int n_y = 10;

		// massimo di righe del dataset che legge
		const int training_num = 60000;

		// massimo di immagini che usa per il train per ogni numero
		public static int max_img = 6000; // 100

		// numero totale di layer
		public static Layer[] hidden_layer = new Layer[(n_hidden_layer-1)+2];
		//public static Layer output_layer = new Layer(n_neuron_hidden, n_y);

		//public static string[] training_input = {"0","1","2"/*,"3",/*"4","5"*/};
		//public static string[] training_target = { { 0, 0, 1 }, { 0, 1, 0 }, { 0, 0, 1 } };

		static void Main(string[] args)
		{
			Console.WriteLine ("nuovo metodo");
			load_model ();
			menu ();
		}

		public static void load_model(){
			init_network ();
			StreamReader model = new StreamReader ("model.txt");
			if (new FileInfo ("model.txt").Length > 0) {
				Console.WriteLine ("Carico Modello...");
				string[] all_data = model.ReadLine ().Split (';');
				Console.WriteLine ("Modello Valido! -> " + all_data.Length);

				int c = 0;
				// carico i layer
				for (int i = 0; i < hidden_layer.Length; i++) {
					// ciclo i neuroni del layer
					for (int j = 0; j < hidden_layer [i].neuron.Length; j++) {
						// ciclo i pesi di ogni neurone del layer
						for (int k = 0; k < hidden_layer [i].neuron [j].w.Length; k++) {
							// scrivo nel file il peso 
							hidden_layer [i].neuron [j].w [k] = double.Parse (all_data [c]);
							c++;
						}
						// scrivo nel file il bias
						hidden_layer [i].bias = double.Parse (all_data [c]);
						c++;
					}
				}	
				Console.WriteLine (c);
				Console.WriteLine ("Modello Caricato!");
			} else {
				// inizializza se non ho il file
				Console.WriteLine ("Creo modello...");
				Console.WriteLine (DateTime.Now);
				for (int i = 0; i < hidden_layer.Length; i++) {
					for (int j = 0; j < hidden_layer[i].neuron.Length; j++) {
						hidden_layer [i].neuron [j].init_w ();
					}
					hidden_layer [i].bias = 1;
				}	
				Console.WriteLine (DateTime.Now);
				Console.WriteLine ("Modello Creato!");
			}
			model.Close ();
		}

		public static void save_model(){
			StreamWriter model = new StreamWriter ("model.txt");

			// salvo i layer
			for (int i = 0; i < hidden_layer.Length; i++) {
				// ciclo i neuroni del layer
				for (int j = 0; j < hidden_layer [i].neuron.Length; j++) {
					// ciclo i pesi di ogni neurone del layer
					for (int k = 0; k < hidden_layer [i].neuron [j].w.Length; k++) {
						// scrivo nel file il peso 
						model.Write (hidden_layer [i].neuron [j].w [k] + ";");
					}
					// scrivo nel file il bias
					model.Write (hidden_layer [i].bias + ";");
				}
			}
			model.Close ();
		}

		public static void menu(){
			string scelta;
			Console.WriteLine("\n1)Train\n2)From csv\n3)From test.bmp\n4)Salva modello\n5)Config\n6)Reset Data");
			scelta = Console.ReadLine();

			switch (scelta) {
			case "1":
				train ();
				break;
			case "2":
				test ();
				break;
			case "3":
				readTest ();
				break;
			case "4":
				save_model ();
				break;
			case "5":
				config ();
				break;
			case "6":
				reset_data ();
				break;
			}
			Console.WriteLine ("\n");

			menu ();
		}

		#region RESET DATA

		public static void reset_data(){
			for (int i = 0; i < hidden_layer.Length; i++) {
				for (int j = 0; j < hidden_layer [i].neuron.Length; j++) {
					hidden_layer [i].neuron [j].init_w ();
				}
				hidden_layer [i].bias = 1;
			}	
		}

		#endregion

		#region CHANGE CONFIG

		public static void config(){
			Console.WriteLine ("\nEpoche : ");
			epoche = Convert.ToInt32 (Console.ReadLine ());
			Console.WriteLine ("\nMax img : ");
			max_img = Convert.ToInt32 (Console.ReadLine ());
			Console.WriteLine ("\nEps : ");
			double eps = Convert.ToDouble (Console.ReadLine ());
			for (int i = 0; i < hidden_layer.Length; i++) {
				// ciclo i neuroni del layer
				for (int j = 0; j < hidden_layer [i].neuron.Length; j++) {
					// ciclo i pesi di ogni neurone del layer
					hidden_layer [i].neuron[j].learning_rate = eps;
				}
			}	

			menu ();
		}

		#endregion

		#region INIT THE NETWORK

		public static void init_network(){
			Console.WriteLine ("INIZIALIZZO ... ");
			Console.WriteLine (DateTime.Now);
			hidden_layer[0] = new Layer(n_x, n_neuron_hidden, "sigmoide");
			for(int i = 1; i < (hidden_layer.Length-1)-1; i++)
			{
				hidden_layer[i] = new Layer(n_neuron_hidden, n_neuron_hidden, "sigmoide");
			}
			//layer output
			hidden_layer[hidden_layer.Length-1] = new Layer(n_neuron_hidden, n_y, "sigmoide");
			Console.WriteLine (DateTime.Now);
			Console.WriteLine ("INIZIALIZZAZIONE FINITA!");
		}

		#endregion

		public static void train(){
			//string path = Directory.GetCurrentDirectory ();
			//train;
			Console.WriteLine ("ALLENO..");
			DateTime beforeTrain = DateTime.Now;
			//Console.WriteLine (DateTime.Now);

			double mse = 0;
			int mse_count = 0;

			int correct = 0;
			int incorrect = 0;
			int n_img = 0;

			// ciclo per le epoche
			for (int e = 0; e < epoche; e++) {
				n_img = 0;
				correct = 0;
				incorrect = 0;
				mse = 0; 
				mse_count = 0;
				foreach (var image in MnistReader.ReadTrainingData())
				{
					int index = Convert.ToInt32 (image.Label);
					double[] _out = new double[n_y];	
					_out = new double[n_y];
					for (int k = 0; k < n_y; k++) {
						_out [k] = 0;
					}
					_out [index] = 1;

					double[] rgb = new double[image.Data.Length];
					for (int i = 0; i < image.Data.Length; i++) {
						rgb [i] = Convert.ToDouble(image.Data [i])/255;
					} 
					//Console.Write("Feed Forward -> " + DateTime.Now);
					feed_forward (rgb);	
					//Console.Write(" --- End Feed Forward -> " + DateTime.Now);
					//Console.Write (" --- Backward Propagation -> " + DateTime.Now);
					back_propagation (_out);
					//Console.Write(" --- End Backward Propagation -> " + DateTime.Now + "\n");

					// CALCOLO MSE
					//hidden_layer [hidden_layer.Length-1].output ();
					double num = _out [index] - hidden_layer [hidden_layer.Length-1]._out[index];
					mse += num * num;
					mse_count++;  

					int max_pos = 0;
					for (int i = 1; i < hidden_layer [hidden_layer.Length - 1]._out.Length; i++) {
						if (hidden_layer [hidden_layer.Length - 1]._out [i] > hidden_layer [hidden_layer.Length - 1]._out [max_pos]) {
							max_pos = i;
						}
					}

					if (max_pos == index) correct++;
					else incorrect++;

					Console.WriteLine ("TRAINING :    No. " + n_img + " of 60000 [ " + Convert.ToInt32 ((n_img * 100) / 60000) + "%]  Correct : " + correct + "  Incorrect : " + incorrect + "  Accuracy : " + Math.Round(((1 - (mse / mse_count))*100),4) + "%  MSE : " + Math.Round((mse / mse_count)*100, 4) + "%");
					n_img++;
				}
			}
			DateTime afterTrain = DateTime.Now;

			mse = 0;
			mse_count = 0;

			correct = 0;
			incorrect = 0;
			n_img = 0;

			foreach (var image in MnistReader.ReadTestData())
			{
				int index = Convert.ToInt32 (image.Label);
				double[] _out = new double[n_y];	
				_out = new double[n_y];
				for (int k = 0; k < n_y; k++) {
					_out [k] = 0;
				}
				_out [index] = 1;

				double[] rgb = new double[image.Data.Length];
				for (int i = 0; i < image.Data.Length; i++) {
					rgb [i] = Convert.ToDouble(image.Data [i])/255;
				} 

				feed_forward (rgb);	

				// CALCOLO MSE
				//hidden_layer [hidden_layer.Length-1].output ();
				double num = _out [index] - hidden_layer [hidden_layer.Length-1]._out[index];
				mse += num * num;
				mse_count++;  

				int max_pos = 0;
				for (int i = 1; i < hidden_layer [hidden_layer.Length - 1]._out.Length; i++) {
					if (hidden_layer [hidden_layer.Length - 1]._out [i] > hidden_layer [hidden_layer.Length - 1]._out [max_pos]) {
						max_pos = i;
					}
				}

				if (max_pos == index) correct++;
				else incorrect++;

				Console.WriteLine ("READING:    No. " + n_img + " of 10000 [ " + Convert.ToInt32 ((n_img * 100) / 10000) + "%]  Correct : " + correct + "  Incorrect : " + incorrect + "  Accuracy : " + Math.Round(((1 - (mse / mse_count))*100),4) + "%  MSE : " + Math.Round((mse / mse_count)*100, 4) + "%");
				n_img++;
			}

			Console.WriteLine ("DONE - Total execution time :  " + (afterTrain - beforeTrain));

			Console.WriteLine ("MSE : " + Math.Round(mse / mse_count,3) + " - mse (total) : " + Math.Round(mse,3) + " - mse_count : " + mse_count);
			//Console.WriteLine (DateTime.Now);

			Console.WriteLine ("Train completato!");
		}


		public static void test(){
			StreamReader test = new StreamReader("testFile.csv");
			string testo = test.ReadLine();
			string[] split = testo.Split(',');
			double[] rgb = new double[split.Length];
			for (int j = 0; j < split.Length; j++) {
				rgb [j] = Convert.ToDouble (split [j])/255;
				//Console.Write(Convert.ToDouble (split [j]));
				if(j < split.Length-1){
					//Console.Write(",");
				}
			}
			//Console.WriteLine();
			test.Close();
			feed_forward(rgb);
			//feed_forward(readTest());
			for (int j = 0; j < n_y; j++) {
				Console.WriteLine("PREDICT : " + j + " -> " + Math.Round(hidden_layer [hidden_layer.Length-1]._out[j]*100,2) + "%");	
			}
			int max_pos = 0;
			double max_out = hidden_layer [hidden_layer.Length-1]._out[0];
			for (int j = 1; j < n_y; j++) {
				if (max_out < hidden_layer [hidden_layer.Length-1]._out[j]) {
					max_pos = j;
					max_out = hidden_layer [hidden_layer.Length-1]._out[j];
				}	
			}
			Console.WriteLine("\nPREDICT : " + max_pos + " -> " + Math.Round(max_out*100,2) + "%");
		}

		public static void readTest(){
			Bitmap img = (Bitmap) Image.FromFile("test.bmp");
			int width = img.Width;
			int height = img.Height;
			double[] rgb = new double[width*height];
			int t = 0;
			for(int i = 0; i < width; i++){
				for(int j = 0; j < height; j++){
					rgb[t] = 255 - img.GetPixel(j,i).R;
					rgb [t] = rgb [t] / 255;
					//Console.Write (rgb [t]*255 + ",");
					t++;
				}
			}
			img.Dispose ();
			feed_forward(rgb);
			//feed_forward(readTest());
			for (int j = 0; j < n_y; j++) {
				Console.WriteLine("PREDICT : " + j + " -> " + Math.Round(hidden_layer [hidden_layer.Length-1]._out[j]*100,2) + "%");	
			}
			int max_pos = 0;
			double max_out = hidden_layer [hidden_layer.Length-1]._out[0];
			for (int j = 1; j < n_y; j++) {
				if (max_out < hidden_layer [hidden_layer.Length-1]._out[j]) {
					max_pos = j;
					max_out = hidden_layer [hidden_layer.Length-1]._out[j];
				}	
			}
			Console.WriteLine("\nPREDICT : " + max_pos + " -> " + Math.Round(max_out*100,2) + "%");
			//mse ();
		}

		public static void back_propagation(double[] target){
			// NON TOCCARE PER NESSUN MOTIVO

			// layer output
			for (int i = 0; i < hidden_layer [hidden_layer.Length-1].neuron.Length; i++) {
				hidden_layer [hidden_layer.Length-1].neuron [i].update_weights (target [i] - hidden_layer [hidden_layer.Length-1]._out[i], hidden_layer [hidden_layer.Length-1]);
			}

			// layer prima del layer outpu
			for (int l = (hidden_layer.Length -1) - 1; l >= 0; l--) {
				for (int k = 0; k < hidden_layer [l + 1].neuron.Length; k++) {
					for (int j = 0; j < hidden_layer [l + 1].neuron [k].input.Length; j++) {
						hidden_layer [l].neuron [j].update_weights (hidden_layer [l + 1].neuron [k].err * hidden_layer [l + 1].neuron [k].w [j], hidden_layer [l]);	
					}
				}
			}

		}

		public static void feed_forward(double[] _input)
		{
			// primo layer
			for (int i = 0; i < n_neuron_hidden; i++) {
				hidden_layer [0].neuron [i].input = _input;	
				hidden_layer [0].neuron [i].feed_forward (hidden_layer [0]);
			}

			// tutti gli altri layer
			for(int i = 1; i < hidden_layer.Length; i++)
			{
				for (int j = 0; j < hidden_layer[i].neuron.Length; j++) {
					hidden_layer [i].neuron [j].input = hidden_layer [i-1].output ();
					hidden_layer [i].neuron [j].feed_forward (hidden_layer [i]);
				}
			}

			hidden_layer [hidden_layer.Length - 1].output ();
		}
	}
}
