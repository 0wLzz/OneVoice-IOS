#import <Foundation/Foundation.h>

NS_ASSUME_NONNULL_BEGIN

@interface OVInference : NSObject
+ (nullable NSString *)transcribe:(NSData *)pcm
                         language:(NSString *)language
                        resources:(NSURL *)resources
                            error:(NSError **)error;
+ (nullable NSString *)tag:(NSString *)transcript
                 resources:(NSURL *)resources
                     error:(NSError **)error;
+ (nullable NSString *)translate:(NSString *)transcript
                        language:(NSString *)language
                             tag:(NSString *)tag
                       resources:(NSURL *)resources
                           error:(NSError **)error;
@end

NS_ASSUME_NONNULL_END
