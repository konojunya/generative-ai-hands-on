import { printFilesize, printTimestamp } from "@/lib/utils";
import { Avatar, AvatarFallback, AvatarImage } from "./ui/avatar";
import { FaFile, FaFolder } from "react-icons/fa";
import { useRouter } from "next/navigation";
import Link from "next/link";

type Item = {
  id: string;
  name: string;
  timestamp: Date;
  isFolder: boolean;
  size: number;
  url: string;
};

const SingleItem = ({ id, name, timestamp, isFolder, size, url }: Item) => {
  const router = useRouter();
  const getUrl = () => {
    if (isFolder) return `/folders/${id}`;
    return url ? url : "";
  };

  const renderInnerContent = () => (
    <div className="hover:bg-slate-100 flex flex-row items-center w-full pr-[2px] h-12 border-b border-solid border-b-[#dadce0] sticky gap-x-2 cursor-pointer">
      <div className="grow min-w-[200px] text-[14px] truncate flex items-center gap-3">
        <div className="pl-2">
          {isFolder ? <FaFolder size={24} /> : <FaFile size={24} />}
        </div>
        <span>{name}</span>
      </div>
      <div className="flex-none w-[140px] text-[14px]">
        <div className="flex items-center gap-2">
          <Avatar className="w-[24px] h-[24px]">
            <AvatarImage
              className="w-full h-full"
              src="/images/placeholder.png"
              alt="avatar"
            />
            <AvatarFallback>CN</AvatarFallback>
          </Avatar>
          <span>自分</span>
        </div>
      </div>
      <div className="flex-none w-[156px] text-[14px]">
        {printTimestamp(timestamp)}
      </div>
      <div className="flex-none w-[156px] text-[14px]">
        {isFolder ? "-" : printFilesize(size)}
      </div>
      <div className="flex-none w-[16px] text-[14px]">︙</div>
    </div>
  );

  return (
    <div className="flex flex-col pl-5 pr-3">
      {isFolder ? (
        <Link href={getUrl()}>{renderInnerContent()}</Link>
      ) : (
        <a href={getUrl()} target="_blank">
          {renderInnerContent()}
        </a>
      )}
    </div>
  );
};

export default SingleItem;
