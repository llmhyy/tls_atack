import javax.net.ssl.SSLSocket;
import javax.net.ssl.SSLSocketFactory;
import java.io.IOException;
import java.io.InputStream;
import java.io.OutputStream;
import java.net.InetSocketAddress;
import java.nio.charset.StandardCharsets;
import java.security.Security;

public class main {
	public static void main(String[] args) {
		//Enable use of sslv3
		Security.setProperty("jdk.tls.disabledAlgorithms", "");
		
		SSLSocketFactory factory = (SSLSocketFactory) SSLSocketFactory.getDefault();
		SSLSocket soc = null;
		try {
			soc = (SSLSocket) factory.createSocket();
		} catch (IOException e) {
			e.printStackTrace();
		}
		
		String ip = args[0];
		int port = Integer.parseInt(args[1]);
		
		String proxyIP = "192.168.1.197";
		int proxyPort = 8080;
		
		try {
			//String[] protocols = {"SSLv3"};
			//soc.setEnabledProtocols(protocols);
			soc.connect(new InetSocketAddress(proxyIP, proxyPort));
			
			soc.startHandshake();
			//soc.setSoTimeout(5000);
			OutputStream stream = soc.getOutputStream();
			InputStream inputStream = soc.getInputStream();
			String s = "GET / HTTP/1.1\n" +
					"Host: forums.cpanel.net\n" +
					"User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:63.0) Gecko/20100101 Firefox/63.0\n" +
					"Accept: text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8\n" +
					"Accept-Language: en-US,en;q=0.5\n" +
					"Accept-Encoding: gzip, deflate\n" +
					"DNT: 1\n" +
					"Connection: keep-alive\n" +
					"Upgrade-Insecure-Requests: 1\n" +
					"Cache-Control: max-age=0\r\n\r\n";
			s = "abc\r\n";
			System.out.println("Begin send data...");
			stream.write(s.getBytes(StandardCharsets.UTF_8));
			System.out.println("Begin received...");
			byte[] data = inputStream.readAllBytes();
			System.out.println(new String(data));
			
			//convert to proxy
			int i = 0;
			while (i++ < 10) {
				soc.connect(new InetSocketAddress(proxyIP, proxyPort));
				System.out.println("Begin send data...");
				stream.write(s.getBytes(StandardCharsets.UTF_8));
				System.out.println("Begin received...");
				data = inputStream.readAllBytes();
				System.out.println(new String(data));
			}
		} catch (IOException e) {
			e.printStackTrace();
		}
		
	}
}