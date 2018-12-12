no_of_data_bits = 64;%bits per channel
M =4;                %Number of subcarrier channels
n=256;               %Total number of bits to be transmitted at the transmitter
block_size = 16;     %Size of each OFDM block to add cyclic prefix
cp_len = floor(0.2 * block_size); %Length of the cyclic prefix
% Generate random data source to be transmitted of length no_of_data_bits
data = randsrc(1, no_of_data_bits, 0:M-1);
% QPSK modulation
qpsk_modulated_data = pskmod(data, M);

% dividing up the data into the carriers
number_of_subcarriers=4;
S2P = reshape(qpsk_modulated_data, no_of_data_bits/number_of_subcarriers,number_of_subcarriers);
Sub_carrier1 = S2P(:,1);
Sub_carrier2 = S2P(:,2);
Sub_carrier3 = S2P(:,3);
Sub_carrier4 = S2P(:,4);

% IFFT block
cp_start=block_size-cp_len;

% Adding cyclic prefixes
for i=1:number_of_subcarriers,
    display("IFFT");
    ifft_Subcarrier(:,i) = ifft((S2P(:,i)),16);% 16 is the ifft point
    disp(ifft_Subcarrier(:,i));
    for j=1:cp_len,
        
        cyclic_prefix(j,i) = ifft_Subcarrier(j+cp_start,i);
    end
    display("Cyclic Prefix");
    disp(cyclic_prefix(1:cp_len,i));
    display("Append Prefix");
    Append_prefix(:,i) = vertcat( cyclic_prefix(:,i), ifft_Subcarrier(:,i));
    disp(Append_prefix(:,i));
end

% Appends prefix to each subcarriers
A1=Append_prefix(:,1);
A2=Append_prefix(:,2);
A3=Append_prefix(:,3);
A4=Append_prefix(:,4);

%Convert to serial stream for transmission
[rows_Append_prefix, cols_Append_prefix]=size(Append_prefix);
len_ofdm_data = rows_Append_prefix*cols_Append_prefix;
% OFDM signal to be transmitted
ofdm_signal = reshape(Append_prefix, 1, len_ofdm_data);

% Adding channel noise
channel = randn(1,2) + sqrt(-1)*randn(1,2);
after_channel = filter(channel, 1, ofdm_signal);
awgn_noise = awgn(zeros(1,length(after_channel)),0);
recvd_signal = awgn_noise+after_channel; % With AWGN noise

for i = 1:76
    fprintf('{%f, %f}, ',real(ofdm_signal(1,i)),imag(ofdm_signal(1,i)));
end
